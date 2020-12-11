#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 19:47:36 2020

@author: ghiggi
"""

import os
import glob 
import tempfile
import re
import numpy as np
import xarray as xr 
from scipy.spatial import SphericalVoronoi

#-----------------------------------------------------------------------------.
###########################
### Polygon formatting ####
###########################
# In future, this section will be moved to a future PolyConversion.py file 
# --> PolyConversion class in development 
# --> Useful for plotting 
### TODO: from lon_bnds, lat_bnds --> list_polygons_latlon

def get_lat_lon_bnds(list_poly_lonlat):
    """
    Reshape a list of polygons in lat_bnds, lon_bnds array.
    
    Outputs arrays format: 
    - Each row represent a polygon 
    - Each column represent a polygon vertex coordinate
    
    The polygon with the largest number of vertices defines the number of columns of the matrices
    For polygons with less vertices, the last vertex coordinates is repeated.

    Parameters
    ----------
    list_poly_lonlat : list
        List of numpy.ndarray with the polygon mesh vertices for each graph node.

    Returns
    -------
    lon_bnds : numpy.ndarray
        Array with the longitude vertices of each polygon.
    lat_bnds : numpy.ndarray
         Array with the latitude vertices of each polygon.

    """
    # Retrieve max number of vertex    
    n_max_vertex = 0
    for p_lonlat in list_poly_lonlat:
        if (len(p_lonlat) > n_max_vertex):
            n_max_vertex = len(p_lonlat)
    # Create polygon x vertices coordinates arrays
    n_poly = len(list_poly_lonlat)
    lat_bnds = np.empty(shape=(n_poly,n_max_vertex)) * np.nan  
    lon_bnds = np.empty(shape=(n_poly,n_max_vertex)) * np.nan 
    for i, p_lonlat in enumerate(list_poly_lonlat):
        tmp_lons = p_lonlat[:,0]
        tmp_lats = p_lonlat[:,1]
        if (len(tmp_lats) < n_max_vertex): # Repeat the last vertex to have n_max_vertex values  
            for _ in range(n_max_vertex - len(tmp_lats)):
                tmp_lons = np.append(tmp_lons, tmp_lons[-1])
                tmp_lats = np.append(tmp_lats, tmp_lats[-1])
        lat_bnds[i,:] = tmp_lats.tolist()
        lon_bnds[i,:] = tmp_lons.tolist()
    return lon_bnds, lat_bnds

#-----------------------------------------------------------------------------.
###############################
### Coordinates conversion ####
###############################
def lonlat2xyz(longitude, latitude, radius=6371.0e6):
    """From 2D geographic coordinates to cartesian geocentric coordinates."""
    ## - lat = phi 
    ## - lon = theta
    ## - radius = rho 
    lon, lat = np.deg2rad(longitude), np.deg2rad(latitude)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z

def xyz2lonlat(x,y,z, radius=6371.0e6):
    """From cartesian geocentric coordinates to 2D geographic coordinates."""
    latitude = np.arcsin(z / radius)/np.pi*180
    longitude = np.arctan2(y, x)/np.pi*180
    return longitude, latitude 

def xyz2polar(x,y,z):
    """From cartesian geocentric coordinates to spherical polar coordinates."""
    r = np.sqrt(x*x + y*y + z*z)  
    theta = np.arccos(z/r) 
    phi = np.arctan2(y, x)
    return r, theta, phi 

#-------------------------------------------------------------------------.
##############################
### Mesh generation tools ####
##############################
 
def SphericalVoronoiMesh_from_lonlat_coords(lon, lat):
    """
    Infer the mesh of a spherical sampling from the mesh node centers provided in 2D geographic coordinates.
    
    Parameters
    ----------
    lon : numpy.ndarray
        Array of longitude coordinates (in degree units).
    lat : numpy.ndarray
        Array of latitude coordinates (in degree units).

    Returns
    -------
    list_polygons_lonlat : list
        List of numpy.ndarray with the polygon mesh vertices for each graph node.

    """
    # Convert to geocentric coordinates 
    radius = 6371.0e6 # radius = 1 can also be used
    x, y, z = lonlat2xyz(lon, lat, radius=radius)
    coords = np.column_stack((x,y,z))
    # Apply Spherical Voronoi tesselation
    sv = SphericalVoronoi(coords,
                          radius=radius, 
                          center=[0, 0, 0])
    #-------------------------------------------------------------------------.
    # SphericalVoronoi object methods 
    # - sv.vertices : Vertex coords
    # - sv.regions : Vertex ID of each polygon
    # - sv.sort_vertices_of_regions() : sort indices of vertices to be clockwise ordered
    # - sv.calculate_areas() : compute the area of the spherical polygons 
    #-------------------------------------------------------------------------.
    # Sort vertices indexes to be clockwise ordered
    sv.sort_vertices_of_regions()
    #-------------------------------------------------------------------------.
    # Retrieve list of polygons coordinates arrays
    list_polygons_lonlat = []
    for region in sv.regions:  
        tmp_xyz = sv.vertices[region]    
        tmp_lon, tmp_lat = xyz2lonlat(tmp_xyz[:,0],tmp_xyz[:,1],tmp_xyz[:,2], radius=radius)    
        list_polygons_lonlat.append(np.column_stack((tmp_lon, tmp_lat)))
    #-------------------------------------------------------------------------.
    return list_polygons_lonlat

def SphericalVoronoiMesh_from_pygsp(graph):
    """
    Compute the mesh of a pygsp spherical graph using Spherical Voronoi.

    Parameters
    ----------
    graph : pgysp.graphs.nngraphs.sphere*
        pgysp graph of a spherical sampling.

    Returns
    -------
    list_polygons_lonlat : List
         List of numpy.ndarray with the polygon mesh vertices for each graph node.
    
    """
    radius = 1
    graph.set_coordinates('sphere', dim=3)
    sv = SphericalVoronoi(graph.coords, radius=radius, center=[0, 0, 0])   
    sv.sort_vertices_of_regions()
    #-------------------------------------------------------------------------.
    # Retrieve list of polygons coordinates arrays
    list_polygons_lonlat = []
    for region in sv.regions:  
        tmp_xyz = sv.vertices[region]    
        tmp_lon, tmp_lat = xyz2lonlat(tmp_xyz[:,0],tmp_xyz[:,1],tmp_xyz[:,2], radius=radius)    
        list_polygons_lonlat.append(np.column_stack((tmp_lon, tmp_lat)))
    #-------------------------------------------------------------------------.
    return list_polygons_lonlat

#-----------------------------------------------------------------------------.
############################
### MPAS & ECMWF grids  ####
############################

def create_ECMWF_atlas_mesh(ECMWF_grid_name, ECMWF_atlas_mesh_dir):
    """
    Generate ECMWF grids using (pre-installed) atlas-meshgen.

    Parameters
    ----------
    ECMWF_grid_name : str
        ECWMF grid name. Example: N320, O640, F1280.
    ECMWF_atlas_mesh_dir : str
        Directory where to save the atlas generated mesh.

    """
    atlas_mesh_fname = "".join([ECMWF_grid_name,".msh"])
    atlas_mesh_fpath = os.path.join(ECMWF_atlas_mesh_dir, atlas_mesh_fname)
    cmd = "".join(["atlas-meshgen ",
                   ECMWF_grid_name, " ",
                   atlas_mesh_fpath, " ",
                   "--lonlat --generator=delaunay"])   
    out = os.system(cmd)
    return out

##----------------------------------------------------------------------------.
def read_ECMWF_atlas_msh(fpath):
    """
    Read the ECMWF Gaussian Grid Specification .msh file generated by atlas-meshgen.
    
    More info at: 
    - https://sites.ecmwf.int/docs/atlas/
    - https://github.com/ecmwf/atlas
    
    Parameters
    ----------
    fpath : str
        Filepath of a ECWMF 2D Mesh Specification .msh file generated by atlas-meshgen.

    Returns
    -------
    lon : numpy.ndarray
        The longitude center of each grid cell.
    lat : numpy.ndarray
       The latitude center of each grid cell.
    list_polygons_latlon : list
       List of numpy.ndarray vertex coordinates with dimension (n_vertex, 2)         
    """
    #-------------------------------------------------------------------------.
    ### msh file format 
    # ...
    # $Nodes
    # 23112
    # 1 0 88.9277 0
    # 20 342 88.9277 0
    # ...
    # $EndNodes
    # $Elements 
    # 30454
    # 1 3 4 1 1 1 0 966 1086 1087 967 # 4 vertex 
    # id 3 4 1 1 1 0 967 1087 1088 968
    # 15767 2 4 1 1 1 0 1 21 22   # 3 vertex
    # 15768 2 4 1 1 1 0 1 22 2    
    # $EndElements
    ##-------------------------------------------------------------------------.
    #f = open(fpath)
    with open(fpath) as f:
        if (f.readline() != '$MeshFormat\n'):
            raise ValueError("Not the expected .msh file format")
        # Dummy lines
        f.readline() # '2.2 0 8\n'
        f.readline() # '$EndMeshFormat\n'
        f.readline() # '$Nodes\n'
        # Retrieve number of cells
        n_cells = int(f.readline().rstrip()) # '23240\n'
        # Initialize lat_vertex_dict, lon_vertex_dict
        lat_cells_dict = {}
        lon_cells_dict = {}
        ##---------------------------------------------------------------------.
        # Retrieve lon lat index of each cell
        for _ in range(n_cells):
            tmp = f.readline()       # '1 0 88.9277 0\n'
            tmp_split = tmp.split(" ") # ['1', '0', '88.9277', '0\n']
            lon_cells_dict[tmp_split[0]] = float(tmp_split[1]) 
            lat_cells_dict[tmp_split[0]] = float(tmp_split[2])
        # Dummy lines
        f.readline() # '$EndNodes\n'
        f.readline() # '$Elements\n'
        ##---------------------------------------------------------------------.
        # Retrieve cell neighbors of each vertex
        n_vertex = int(f.readline().rstrip()) # '30418\n'
        vertex_cells_dict = {}
        for _ in range(n_vertex):
            tmp = f.readline() # '1 3 4 1 1 1 0 966 1086 1087 967\n'
            tmp = tmp.rstrip() # remove \n
            tmp_split = tmp.split(" ") # ['1', '3', '4', '1', '1', '1', '0', '966', '1086', '1087', '967']
            tmp_id_vertex = tmp_split[7:]
            vertex_cells_dict[tmp_split[0]] = tmp_id_vertex
        ##---------------------------------------------------------------------.
        # Dummy line     
        if (f.readline() != '$EndElements\n'):
            raise ValueError("Something went wrong in parsing the .msh file")
    ##-------------------------------------------------------------------------.   
    # Retrieve lon, lat arrays 
    lon = np.array(list(lon_cells_dict.values()))
    lat = np.array(list(lat_cells_dict.values()))
    ##-------------------------------------------------------------------------.
    # Retrieve list of mesh polygons (based on $Elements indexing)
    n_vertex = len(vertex_cells_dict.keys())
    list_polygons_latlon = [] 
    for i, cellIDs in enumerate(vertex_cells_dict.values()):
        # Retrieve lon lat of neigbor nodes 
        tmp_lats = [lat_cells_dict[i] for i in cellIDs] 
        tmp_lons = [lon_cells_dict[i] for i in cellIDs] 
        tmp_lonlat = np.column_stack((tmp_lons, tmp_lats))
        list_polygons_latlon.append(tmp_lonlat)
    ##-------------------------------------------------------------------------.        
    return (lon, lat, list_polygons_latlon)

###---------------------------------------------------------------------------.
def read_MPAS_mesh(fpath):
    """
    Read the MPAS Mesh Specification netCDF4 file and returns grid centers and corners.
    
    MPAS meshes can be downloaded at https://mpas-dev.github.io/atmosphere/atmosphere_meshes.html .

    Parameters
    ----------
    fpath : str
        Filepath of a MPAS Mesh Specification netCDF4 file

    Returns
    -------
    lon : numpy.ndarray
        The longitude center of each grid cell.
    lat : numpy.ndarray
       The latitude center of each grid cell.
    lon_bnds : numpy.ndarray
        The longitudes of the corners of each grid cell.
    lat_bnds : numpy.ndarray
        The latitudes of the corners of each grid cell.
        
    """
    ds = xr.load_dataset(fpath)
    ## Dataset Summary
    # ds.latCell  # in radians
    # ds.lonCell  # in radians
    # ds.latVertex.shape
    # ds.lonVertex.shape
    ## verticesOnCell contains 0 when not 6 vertices 
    # - shape: ncells x vertices_idx
    # idxs, counts = np.unique(ds.verticesOnCell.values, return_counts=True)
    # counts
    #-------------------------------------------------------------------------.
    # Convert cell centers to degrees 
    lat = ds.latCell.values * 180 / np.pi
    lon = ds.lonCell.values * 180 / np.pi  
    ##------------------------------------------------------------------------.
    # Preprocess indexes for cells with less than X vertices 
    idx_vertices = ds.verticesOnCell.values
    # Drop columns with only 0 
    idx_vertices = idx_vertices[:, idx_vertices.sum(axis=0) != 0]
    # - Replace 0 with idx of previous vertex 
    row_idx, col_idx = np.where(idx_vertices == 0)
    for i in range(len(col_idx)):
        idx_vertices[row_idx[i],col_idx[i]] = idx_vertices[row_idx[i],col_idx[i]-1]
    # Subtract 1 for python index based on 0 
    idx_vertices = idx_vertices - 1   
    # Retrieve lat lon of vertices 
    lat_bnds = ds.latVertex.values[idx_vertices] * 180 / np.pi
    lon_bnds = ds.lonVertex.values[idx_vertices] * 180 / np.pi 
    ##------------------------------------------------------------------------.
    return (lon, lat, lon_bnds, lat_bnds)

#-----------------------------------------------------------------------------.
############################ 
### CDO remapping tools ####
############################ 
###----------------------------------------------------------------------------.
# Utils function used to write CDO grid files 
def arr2str(arr):
    """Convert numpy 1D array into single string of spaced numbers."""
    return "  ".join(map(str, list(arr)))

###----------------------------------------------------------------------------.
def write_cdo_grid(fpath,
                   xvals, 
                   yvals, 
                   xbounds,
                   ybounds):
    """
    Create the CDO Grid Description File of an unstructured grid.
        
    Parameters
    ----------
    fpath : str
        CDO Grid Description File name/path to write
    xvals : numpy.ndarray
        The longitude center of each grid cell.
    yvals : numpy.ndarray
       The latitude center of each grid cell.
    xbounds : numpy.ndarray
        The longitudes of the corners of each grid cell.
    ybounds : numpy.ndarray
        The latitudes of the corners of each grid cell.

    Returns
    -------
    None.

    """
    # Checks 
    if (not isinstance(yvals, np.ndarray)):
        raise TypeError("Provide yvals as numpy.ndarray")
    if (not isinstance(xvals, np.ndarray)):
        raise TypeError("Provide xvals as numpy.ndarray")
    if (not isinstance(xbounds, np.ndarray)):
        raise TypeError("Provide xbounds as numpy.ndarray")
    if (not isinstance(ybounds, np.ndarray)):
        raise TypeError("Provide ybounds as numpy.ndarray")
    if (len(yvals) != len(xvals)):
        raise ValueError("xvals and yvals must have same size") 
    if (ybounds.shape[0] != xbounds.shape[0]):
        raise ValueError("xbounds and ybounds must have same shape")    
    ##------------------------------------------------------------------------. 
    # Retrieve number of patch and max number of vertex
    n_cells = len(yvals)
    nvertex = ybounds.shape[1]
    # Write down the gridType
    with open(fpath, "w") as txt_file:
        txt_file.write("gridtype  = unstructured \n")
        txt_file.write("gridsize  = %s \n" %(n_cells))
        txt_file.write("nvertex   = %s \n" %(nvertex))
        # Write xvals 
        txt_file.write("xvals     = %s \n" %(arr2str(xvals)))
        # Write yvals 
        txt_file.write("yvals     = %s \n" %(arr2str(yvals)))
        # Write xbounds 
        txt_file.write("xbounds   = %s \n" %(arr2str(xbounds[0,:])))
        for line in xbounds[1:,:]:
            txt_file.write("            %s \n" %(arr2str(line)))
        # Write ybounds 
        txt_file.write("ybounds   = %s \n" %(arr2str(ybounds[0,:])))
        for line in ybounds[1:,:]:
            txt_file.write("            %s \n" %(arr2str(line)))               
                           
    print(fpath, "cdo grid written successfully!")  
def get_available_interp_methods(): 
    """Available interpolation methods."""
    methods = ['nearest_neighbors',
               'idw',
               'bilinear',
               'bicubic',
               'conservative',
               'conservative2',
               'largest_area_fraction']
    return methods 

def check_interp_method(method): 
    """Check if interpolation method is valid."""
    if not isinstance(method, str):
        raise TypeError("Provide interpolation 'method' name as a string")
    if (method not in get_available_interp_methods()): 
        raise ValueError("Provide valid interpolation method. get_available_interp_methods()")

def check_normalization(normalization):
    """Check normalization option for CDO conservative remapping."""
    if not isinstance(normalization, str):
        raise TypeError("Provide 'normalization' type as a string")
    if normalization not in ['fracarea','destarea']:
        raise ValueError("Normalization must be either 'fracarea' or 'destarea'")
        
def get_cdo_genweights_cmd(method): 
    """Define available methods to generate interpolation weights in CDO."""
    d = {'nearest_neighbors': 'gennn',
         'idw': 'gendis',
         'bilinear': 'genbil',
         'bicubic': 'genbic', 
         'conservative': 'genycon',
         'conservative2': 'genycon2',  
         'largest_area_fraction': 'genlaf'} 
    return d[method]

def get_cdo_remap_cmd(method): 
    """Define available interpolation methods in CDO."""
    # REMAPDIS - IDW using the 4 nearest neighbors   
    d = {'nearest_neighbors': 'remapnn',
         'idw': 'remapdis',
         'bilinear': 'remapbil',
         'bicubic': 'remapbic', 
         'conservative': 'remapycon',
         'conservative2': 'remapycon2',  
         'largest_area_fraction': 'remaplaf'} 
    return d[method]    

def cdo_genweights(method, 
                   input_grid_path,
                   output_grid_path,
                   input_fpath,
                   interpolation_weights_path, 
                   normalization = 'fracarea',
                   n_threads = 1):
    """
    Wrap around CDO gen* to compute interpolation weights.

    Parameters
    ----------
    method : str
        Interpolation method. 
    input_grid_path : str
        File (path) specifying the grid structure of input data.
    output_grid_path : str
        File (path) specifying the grid structure of output data.
    input_fpath : str
        Filepath of the input file  
    interpolation_weights_path : str 
        Filepath of the CDO interpolation weights.  
    normalization : str, optional
        Normalization option for conservative remapping. 
        The default is 'fracarea'.
        Options:
        - fracarea uses the sum of the non-masked source cell intersected 
          areas to normalize each target cell field value. 
          Flux is not locally conserved.
        - destarea’ uses the total target cell area to normalize each target
          cell field value. 
          Local flux conservation is ensured, but unreasonable flux values 
          may result [i.e. in small patches]. 
    compression_level : int, optional
        Compression level of output netCDF4. Default 1. 0 for no compression.
    n_threads : int, optional
        Number of OpenMP threads to use within CDO. The default is 1.

    Returns
    -------
    None.

    """   
    # TODO: to generalize to whatever case, input_grid_path and -setgrid faculative if real data ..
    # TODO: Check boolean arguments
    ##------------------------------------------------------------------------.  
    # Check arguments 
    check_normalization(normalization)
    # Check that the folder where to save the weights it exists 
    if not os.path.exists(os.path.dirname(interpolation_weights_path)):
        raise ValueError("The directory where to store the interpolation weights do not exists")             
    ##------------------------------------------------------------------------.    
    ## Define CDO options for interpolation (to be exported in the environment)
    # - cdo_grid_search_radius
    # - remap_extrapolate 
    opt_CDO_environment = "".join(["CDO_REMAP_NORM",
                                   "=",
                                   "'", normalization, "'"
                                   "; "
                                   "export CDO_REMAP_NORM; "])
    ##------------------------------------------------------------------------.
    ### Define CDO OpenMP threads options 
    if (n_threads > 1):
        opt_CDO_parallelism = "--worker %s -P %s" %(n_threads, n_threads)  
        # opt_CDO_parallelism = "-P %s" %(n_threads) # necessary for CDO < 1.9.8
    else: 
        opt_CDO_parallelism = ""
    ##------------------------------------------------------------------------.
    ### Compute weights        
    cdo_genweights_command = get_cdo_genweights_cmd(method=method)
    # Define command 
    command = "" .join([opt_CDO_environment,
                        "cdo ",
                        opt_CDO_parallelism, " ", 
                        "-b 64", " ", # output precision
                        cdo_genweights_command, ",",
                        output_grid_path, " ",
                        "-setgrid,",
                        input_grid_path, " ", 
                        input_fpath, " ",
                        interpolation_weights_path])    
    # Run command
    os.system(command) 

def cdo_remapping(method, 
                  input_grid_path,
                  output_grid_path,
                  input_fpaths,
                  output_fpaths,
                  precompute_weights = True,
                  interpolation_weights_path = None, 
                  normalization = 'fracarea',
                  compression_level = 1,
                  n_threads = 1):
    """
    Wrap around CDO to remap grib files to whatever unstructured grid.

    Parameters
    ----------
    method : str
        Interpolation method. 
    input_grid_path : str
        File (path) specifying the grid structure of input data.
    output_grid_path : str
        File (path) specifying the grid structure of output data.
    input_fpaths : list
        Filepaths list of input data to remap.
    output_fpaths : list
        Filepaths list of output data.
    precompute_weights : bool, optional
        Whether to use or first precompute the interpolation weights.
        The default is True.
    interpolation_weights_path : str, optional
        Filepath of the CDO interpolation weights.  
        It is used only if precompute_weights is True. 
        If not specified, it save the interpolation weights in a temporary 
        folder which is deleted when processing ends.        
    normalization : str, optional
        Normalization option for conservative remapping. 
        The default is 'fracarea'.
        Options:
        - fracarea uses the sum of the non-masked source cell intersected 
          areas to normalize each target cell field value. 
          Flux is not locally conserved.
        - destarea’ uses the total target cell area to normalize each target
          cell field value. 
          Local flux conservation is ensured, but unreasonable flux values 
          may result [i.e. in small patches]. 
    compression_level : int, optional
        Compression level of output netCDF4. Default 1. 0 for no compression.
    n_threads : int, optional
        Number of OpenMP threads to use within CDO. The default is 1.

    Returns
    -------
    None.

    """ 
    # TODO: to generalize to whatever case, input_grid_path and -setgrid faculative if real data ... 
    ##------------------------------------------------------------------------.  
    # Check arguments 
    check_normalization(normalization)
    # Define temporary path for weights if weights are precomputed 
    FLAG_temporary_weight = False
    if (precompute_weights is True): 
        FLAG_temporary_weight = False
        if (interpolation_weights_path is None):
            # Create temporary directory where to store interpolation weights 
            FLAG_temporary_weight = True
            interpolation_weights_path = tempfile.NamedTemporaryFile().name
        else: 
            # Check that the folder where to save the weights it exists 
            if not os.path.exists(os.path.dirname(interpolation_weights_path)):
                raise ValueError("The directory where to store the interpolation weights do not exists")             
    ##------------------------------------------------------------------------.    
    ## Define CDO options for interpolation (to export in the environment)
    # - cdo_grid_search_radius
    # - remap_extrapolate 
    opt_CDO_environment = "".join(["CDO_REMAP_NORM",
                                   "=",
                                   "'", normalization, "'"
                                   "; "
                                   "export CDO_REMAP_NORM; "])
    ##------------------------------------------------------------------------.
    ### Define CDO OpenMP threads options 
    if (n_threads > 1):
        opt_CDO_parallelism = "--worker %s -P %s" %(n_threads, n_threads)  
        # opt_CDO_parallelism = "-P %s" %(n_threads) # necessary for CDO < 1.9.8
    else: 
        opt_CDO_parallelism = ""
    ##------------------------------------------------------------------------.
    ### Define netCDF4 compression options 
    if (compression_level > 9 or compression_level < 1):
        opt_CDO_data_compression = "-z zip_%s" %(int(compression_level))
    else: 
        opt_CDO_data_compression = ""
    ##------------------------------------------------------------------------.
    ### Precompute the weights (once) and then remap 
    if (precompute_weights is True): 
        ##--------------------------------------------------------------------.  
        # If weights are not yet pre-computed, compute it 
        if not os.path.exists(interpolation_weights_path):
            cdo_genweights_command = get_cdo_genweights_cmd(method=method)
            # Define command 
            command = "".join([opt_CDO_environment,
                               "cdo ",
                               opt_CDO_parallelism, " ", 
                               "-b 64", " ", # output precision
                               cdo_genweights_command, ",",
                               output_grid_path, " ",
                               "-setgrid,",
                               input_grid_path, " ", 
                               input_fpaths[0], " ",
                               interpolation_weights_path])    
            # Run command
            os.system(command) 
        ##--------------------------------------------------------------------.       
        # Remap all files 
        for input_fpath, output_fpath in zip(input_fpaths, output_fpaths):
            # Define command 
            command = "".join([opt_CDO_environment,
                               "cdo ",
                               opt_CDO_parallelism, " ", 
                               "-b 64", " ", # output precision
                               "-f nc4", " ", # output type: netcdf
                               opt_CDO_data_compression, " ",
                               "remap,",
                               output_grid_path, ",",
                               interpolation_weights_path, " ",
                               "-setgrid,",
                               input_grid_path, " ",
                               input_fpath, " ",
                               output_fpath])
            
            # Run command
            os.system(command)
    ##--------------------------------------------------------------------. 
    ### Remap directly without precomputing the weights
    else: 
        # Retrieve CDO command for direct interpolation
        remapping_command = get_cdo_remap_cmd(method=method)
        # Remap all files 
        for input_fpath, output_fpath in zip(input_fpaths, output_fpaths):
            # Define command 
            command = "".join([opt_CDO_environment,
                               "cdo ",
                               opt_CDO_parallelism, " ", 
                               "-b 64", " ", # output precision
                               "-f nc4", " ",
                               opt_CDO_data_compression, " ",
                               remapping_command, ",",
                               output_grid_path, " ",
                               "-setgrid,",
                               input_grid_path, " ",
                               input_fpath, " ",
                               output_fpath])
            # Run command
            os.system(command)
    #-------------------------------------------------------------------------.
    if (FLAG_temporary_weight is True):
        os.remove(interpolation_weights_path)
    return 

#-----------------------------------------------------------------------------.
#########################################
### pygsp - CDO interpolation weights ###
#########################################   
def _write_dummy_1D_nc(graph):
    """Create a dummy netCDF for CDO based on pygsp graph."""
    # The dummy netCDF is required by CDO to compute the interpolation weights.
    #-------------------------------------------------------------------------.
    # Create dummy filepath 
    fpath = tempfile.NamedTemporaryFile().name
    # Create dummy netCDF  
    n = graph.n_vertices
    data = np.arange(0,n)
    da = xr.DataArray(data = data[np.newaxis], 
                      dims = ['time', 'nodes'],
                      coords = {'time': np.datetime64('2005-02-25')[np.newaxis]},
                      name = 'dummy_var')
    # da.coords["lat"] = ('nodes', data)
    # da.coords["lon"] = ('nodes', data)
    ds = da.to_dataset()
    ds.to_netcdf(fpath)
    return fpath

def pygsp_to_CDO_grid(graph, CDO_grid_fpath):
    """
    Define CDO grid based on pygsp Spherical graph.

    Parameters
    ----------
    graph : pygsp.graph
        pygsp spherical graph.
    CDO_grid_fpath : str
        Filepath where to save the CDO grid.

    Returns
    -------
    None.

    """
    # Check is pygsp graph
    # TODO
    #-------------------------------------------------------------------------.
    # Retrieve graph nodes  
    lon_center = graph.signals['lon']*180/np.pi 
    lat_center = graph.signals['lat']*180/np.pi
    # Enforce longitude between -180 and 180 
    lon_center[lon_center>180] = lon_center[lon_center>180] - 360    
    # Consider it as cell centers and infer it vertex
    list_polygons_lonlat = SphericalVoronoiMesh_from_pygsp(graph) # PolygonArrayList
    # Approximate to 2 digits to define less vertices # TODO ????
    # list_polygons_lonlat1 = []
    # for i, p in enumerate(list_polygons_lonlat):
    #     list_polygons_lonlat1.append(np.unique(np.round(p,2), axis = 0))
    # Reformat polygon vertices array to have all same number of vertices
    lon_vertices, lat_vertices = get_lat_lon_bnds(list_polygons_lonlat)
    # Write down the ECMF mesh into CDO grid format
    write_cdo_grid(fpath = CDO_grid_fpath,
                   xvals = lon_center, 
                   yvals = lat_center, 
                   xbounds = lon_vertices,
                   ybounds = lat_vertices)
    return 

def compute_interpolation_weights(src_graph, dst_graph, 
                                  method = "conservative", 
                                  normalization = 'fracarea',
                                  weights_fpath = None,
                                  CDO_grid_input_fpath = None, 
                                  CDO_grid_output_fpath = None, 
                                  recreate_CDO_grids = False,
                                  return_weights = True): 
    """
    Compute the interpolation weights between two pygsp spherical samplings.
    
    Parameters
    ----------
    src_graph : pygsp.graph
        Source spherical graph.
    dst_graph : pygsp.graph
        Destination spherical graph.
    method : str, optional
        Interpolation/remapping method. The default is "conservative".
    normalization : str, optional
        Normalization option for conservative remapping. 
        The default is 'fracarea'.
        Options:
        - fracarea uses the sum of the non-masked source cell intersected 
          areas to normalize each target cell field value. 
          Flux is not locally conserved.
        - destarea’ uses the total target cell area to normalize each target
          cell field value. 
          Local flux conservation is ensured, but unreasonable flux values 
          may result [i.e. in small patches].
    weights_fpath : str, optional
        Optional filepath where to save the weights netCDF4. The default is None.
        If None, the weights are not saved on disk.
    CDO_grid_input_fpath : str, optional
        Filepath where to save the CDO grid for the source spherical grid. The default is None.
        If None, the CDO grid is not saved on disk.
    CDO_grid_output_fpath : str, optional
        Filepath where to save the CDO grid for the destination spherical grid. The default is None.
        If None, the CDO grid is not saved on disk. 
    recreate_CDO_grids : bool, optional
        Wheter to redefine the CDO grids if CDO_grid_input_fpath or CDO_grid_output_fpath are provided.
        The default is False.
    return_weights : bool, optional
        Wheter to return the interpolation weights. The default is True.

    Returns
    -------
    ds : xarray.Dataset
        Xarray Dataset with the interpolation weights.

    """
    # Check method 
    check_interp_method(method)
    check_normalization(normalization)
    # TODO 
    # Check is pygsp graph
    # Check boolean arguments
    #-------------------------------------------------------------------------.
    # Create temporary fpath if required
    FLAG_tmp_CDO_grid_input_fpath = False
    FLAG_tmp_CDO_grid_output_fpath = False
    FLAG_tmp_weights_fpath = False 
    if CDO_grid_input_fpath is None: 
        FLAG_tmp_CDO_grid_input_fpath = True
        CDO_grid_input_fpath = tempfile.NamedTemporaryFile().name
    if CDO_grid_output_fpath is None: 
        FLAG_tmp_CDO_grid_output_fpath = True
        CDO_grid_output_fpath = tempfile.NamedTemporaryFile().name
    if (weights_fpath is None):
        FLAG_tmp_weights_fpath = True
        weights_fpath = tempfile.NamedTemporaryFile().name 
    #-------------------------------------------------------------------------.
    # Checks if the directory exists 
    if not os.path.exists(os.path.dirname(weights_fpath)):
        raise ValueError("The directory where to store the interpolation weights do not exists.") 
    if not os.path.exists(os.path.dirname(CDO_grid_input_fpath)):
        raise ValueError("The directory where to store the CDO (input) grid do not exists.") 
    if not os.path.exists(os.path.dirname(CDO_grid_output_fpath)):
        raise ValueError("The directory where to store the CDO (output) grid do not exists.")      
    #-------------------------------------------------------------------------.
    # Define CDO grids based on pygsp graph if required 
    if ((recreate_CDO_grids is True) or (FLAG_tmp_CDO_grid_input_fpath is True)):
        pygsp_to_CDO_grid(src_graph, CDO_grid_input_fpath)
    if ((recreate_CDO_grids is True) or (FLAG_tmp_CDO_grid_output_fpath is True)):  
        pygsp_to_CDO_grid(dst_graph, CDO_grid_output_fpath)
    #-------------------------------------------------------------------------.
    # Create a dummy input file for CDO 
    input_fpath = _write_dummy_1D_nc(src_graph)
    #-------------------------------------------------------------------------.
    # Compute interpolation weights 
    cdo_genweights(method, 
                   input_grid_path = CDO_grid_input_fpath,
                   output_grid_path = CDO_grid_output_fpath,
                   input_fpath = input_fpath,
                   interpolation_weights_path = weights_fpath, 
                   normalization = normalization,
                   n_threads = 1)
    #-------------------------------------------------------------------------.
    # Load the weights if required
    if (return_weights):
        ds = xr.open_dataset(weights_fpath)
    #-------------------------------------------------------------------------.
    # Remove dummy files 
    os.remove(input_fpath)
    if FLAG_tmp_weights_fpath:
        os.remove(weights_fpath)
    if FLAG_tmp_CDO_grid_input_fpath:
        os.remove(CDO_grid_input_fpath)
    if FLAG_tmp_CDO_grid_output_fpath:
        os.remove(CDO_grid_output_fpath)
    #-------------------------------------------------------------------------.     
    if return_weights:
        return ds
    else: 
        return 
    
#-----------------------------------------------------------------------------.
########################################
### Ad-hoc funtion for WeatherBench #### 
########################################
### Datasets / Samplings / Variables ####
def get_available_datasets():
    """Available datasets."""
    datasets = ['ERA5_HRES',
                'ERA5_EDA',
                'IFS_HRES',
                'IFS_ENS',
                'IFS_ENS_Extended',
                'SEAS5'] 
    return datasets

def get_native_grids_dict(): 
    """Native grid dictionary of datasets."""
    d = {'ERA5_HRES': 'N320',
         'ERA5_EDA': 'N160', 
         'IFS_HRES': 'O1280',
         'IFS_ENS': 'O640',
         'IFS_ENS_Extended': 'O320',
         'SEAS5': 'O320'}
    return d

def get_native_grid(dataset): 
    """Native grid of a dataset."""
    return get_native_grids_dict()[dataset]
             
def get_available_dynamic_variables():
    """Available dynamic variables."""
    # https://github.com/pangeo-data/WeatherBench 
    variables = ['geopotential',
                 'temperature',
                 'specific_humidity',
                 'toa_incident_solar_radiation']
    return variables        
         
def get_available_static_variables():   
    """Available static variables."""
    variables = ['topography', 'land_sea_mask', 'soil_type']
    return variables 

def get_available_variables():
    """Available variables."""
    variables = get_available_dynamic_variables()
    variables.extend(get_available_static_variables())
    return variables

def get_variable_interp_method_dict(): 
    """Interpolation method dictionary for each variable."""
    d = {'dynamic_variables': 'conservative',
         'topography': 'conservative',
         'land_sea_mask': 'largest_area_fraction',
         'soil_type': 'largest_area_fraction'}
    return d

def get_variable_interp_method(variable):
    """Return the interpolation method that should be used for a specific variable."""
    return get_variable_interp_method_dict()[variable]   

def get_dir_path(data_dir, dataset, sampling, variable_type, variable=None): 
    """Get directory path."""
    dir_path = os.path.join(data_dir, dataset, sampling, variable_type)  
    # Create a subdirectory for each static variable 
    if (variable_type == "static"):
        dir_path = os.path.join(dir_path, variable)   
    return dir_path 

def get_cdo_grid_fpath(CDO_grids_dir, sampling): 
    """Check if CDO grid description file exists and return its path."""
    fpath = os.path.join(CDO_grids_dir, sampling) 
    if not os.path.exists(fpath):
        raise ValueError("Please create a CDO grid description infile into the CDO grids folder")
    return fpath

def get_cdo_weights_filename(method, input_sampling, output_sampling): 
    """Generate the filename where to save the CDO interpolation weights."""
    # Normalization option
    # Nearest_neighbor option 
    filename = "CDO_" + method + "_weights_IN_" + input_sampling + "_OUT_"+ output_sampling + ".nc"
    return filename 

################
### Checks #####
################
def check_dataset(dataset): 
    """Check dataset name."""
    if not isinstance(dataset, str):
        raise TypeError("Provide 'dataset' name as a string")
    if (dataset not in get_available_datasets()): 
        raise ValueError("Provide valid dataset. get_available_datasets()")

def check_sampling(CDO_grids_dir, sampling): 
    """Check sampling name."""
    if not isinstance(sampling, str):
        raise TypeError("Provide 'sampling' name as a string")
    files = os.listdir(CDO_grids_dir)
    if (sampling not in files): 
        raise ValueError("Provide sampling name for which a CDO grid has been defined")

def check_variable(variable):
    """Check variable name."""
    if not isinstance(variable, str):
        raise TypeError("Provide 'variable' name as a string")
    if (variable not in get_available_variables()): 
        raise ValueError("Provide valid variable. get_available_variables()")

def check_variable_type(variable_type):
    """Check variable type name."""
    if not isinstance(variable_type, str):
        raise TypeError("Provide 'variable_type' name as a string")
    if (variable_type not in ['static', 'dynamic']): 
        raise ValueError("Provide either 'static' or 'dynamic'")

def ensure_dir_exists(dir_paths):
    """Create directory if not existing."""
    # Accept str or list of directories.
    if isinstance(dir_paths, str):
        dir_paths = [dir_paths]
    if not isinstance(dir_paths, list):
        raise ValueError('Either provide a string or a list of string')
    # TODO unique dir_path to speed up
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    return 

###---------------------------------------------------------------------------.
### Remapping grib files ####
def change_to_nc_extension(fpath): 
    """Replace .grib with .nc ."""
    pre, ext = os.path.splitext(fpath)
    return pre + ".nc"

def get_input_output_fpaths(input_folder, output_folder):
    """Return input, output tuple of filepaths for a specific input folder."""
    input_fpaths = sorted(glob.glob(input_folder + "/**/*.grib", recursive=True))
    output_fpaths = [re.sub(pattern=input_folder, repl=output_folder, string=fpath) for fpath in input_fpaths]
    # Ensure output dir exists 
    _ = [ensure_dir_exists(os.path.dirname(fpath)) for fpath in output_fpaths]
    # Change extension from grib to nc 
    output_fpaths = [change_to_nc_extension(fpath) for fpath in output_fpaths]
    return (input_fpaths, output_fpaths)

def remap_grib_files(data_dir, 
                     CDO_grids_dir,
                     CDO_grids_weights_dir, 
                     dataset,
                     sampling,
                     variable_type,
                     precompute_weights = True, 
                     normalization='fracarea',
                     n_threads = 1,
                     compression_level = 1, 
                     force_remapping = False): 
    """
    Functions to remap nc/grib files between unstructured grid with cdo.

    Parameters
    ----------
    data_dir : str
        Base directory where data are stored.
    CDO_grids_dir : str
        Directory where CDO grids description files are stored.
    CDO_grids_weights_dir : str
        Directory where the generated weights for interpolation can be stored.
    dataset : str
        A valid dataset name [to access <dataset>/<sampling>/variable_type].
    variable_type : str
        A valid variable type [to access <dataset>/<sampling>/variable_type].
        Either dynamic or static.
    sampling : str
        A valid variable name [to access <dataset>/<sampling>/variable_type].
    precompute_weights : bool, optional
        Whether to first precompute once the iterpolation weights and then remap. 
        The default is True.
    normalization : str, optional
        Normalization option for conservative remapping. 
        The default is 'fracarea'.
        Options:
        - fracarea uses the sum of the non-masked source cell intersected 
          areas to normalize each target cell field value. 
          Flux is not locally conserved.
        - destarea’ uses the total target cell area to normalize each target
          cell field value. 
          Local flux conservation is ensured, but unreasonable flux values 
          may result [i.e. in small patches].
    compression_level : int, optional
        Compression level of output netCDF4. Default 1. 0 for no compression.
    n_threads : int, optional
        Number of OpenMP threads to use within CDO. The default is 1.

    Returns
    -------
    None.

    """
    ##------------------------------------------------------------------------.
    ### Checks 
    check_dataset(dataset)
    check_sampling(CDO_grids_dir, sampling)
    check_variable_type(variable_type)   
    check_normalization(normalization)
    ##------------------------------------------------------------------------.
    ## Define input and output sampling 
    native_sampling = get_native_grid(dataset=dataset)
    # Retrieve the CDO grid description path of inputs and outputs 
    cdo_grid_input_path = get_cdo_grid_fpath(CDO_grids_dir = CDO_grids_dir,
                                            sampling = native_sampling)
    cdo_grid_output_path = get_cdo_grid_fpath(CDO_grids_dir = CDO_grids_dir,
                                             sampling = sampling)
    ##------------------------------------------------------------------------.
    if (variable_type == "static"):
        variables = get_available_static_variables()
    else:  
        variables = ["dynamic_variables"]
    ##------------------------------------------------------------------------.
    for variable in variables:
        print("Remapping", variable, "from", native_sampling, "to", sampling)
        ### Define input and output folders 
        native_folder = get_dir_path(data_dir = data_dir,
                                     dataset = dataset, 
                                     sampling = native_sampling, 
                                     variable_type = variable_type,
                                     variable = variable)
        destination_folder = get_dir_path(data_dir = data_dir,
                                          dataset = dataset, 
                                          sampling = sampling, 
                                          variable_type = variable_type,
                                          variable = variable)
        ##--------------------------------------------------------------------.
        ### List input filepaths and define output filepaths for a specific folder
        input_fpaths, output_fpaths = get_input_output_fpaths(input_folder = native_folder,
                                                              output_folder = destination_folder)
        ##--------------------------------------------------------------------.
        if (len(input_fpaths) == 0):
            print(variable, "data are not available")
            continue
        ##--------------------------------------------------------------------. 
        ## Remap only data not already remapped 
        if force_remapping is not True:
            idx_not_existing = [not os.path.exists(output_fpath) for output_fpath in output_fpaths]
            input_fpaths = np.array(input_fpaths)[np.array(idx_not_existing)].tolist()
            output_fpaths = np.array(output_fpaths)[np.array(idx_not_existing)].tolist() 
        if (len(input_fpaths) == 0):
            print("Data were already remapped. Set force_remapping=True to force remapping.")
            continue    
        ##--------------------------------------------------------------------.
        ### Define interpolation method based on variable_type and variable  
        method = get_variable_interp_method(variable)
        ##--------------------------------------------------------------------. 
        ### Specify filename and path for the interpolation weights  
        cdo_weights_name = get_cdo_weights_filename(method = method,
                                                    input_sampling = native_sampling, 
                                                    output_sampling = sampling)
        interpolation_weights_path = os.path.join(CDO_grids_weights_dir, cdo_weights_name)
        ##--------------------------------------------------------------------. 
        # Remap the data 
        cdo_remapping(method = method,
                      input_grid_path = cdo_grid_input_path,
                      output_grid_path = cdo_grid_output_path, 
                      input_fpaths = input_fpaths,
                      output_fpaths = output_fpaths,
                      precompute_weights = precompute_weights,
                      interpolation_weights_path = interpolation_weights_path, 
                      normalization = normalization,
                      compression_level = compression_level,
                      n_threads = n_threads)
        ##--------------------------------------------------------------------.
    ##-----------------------------------------------------------------------.
    return 

#-----------------------------------------------------------------------------.
