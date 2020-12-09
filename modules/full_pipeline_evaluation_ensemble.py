import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json
import psutil
import healpy as hp
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors


import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter

import xskillscore as xs
from scipy.stats import norm
from xskillscore import crps_quadrature, crps_ensemble, crps_gaussian


import modules.architectures as modelArchitectures
from modules.utils import init_device
from modules.architectures import *
from modules.test import compute_rmse_healpix
from modules.plotting import plot_rmses, plot_rmses_realizations, plot_crps, plot_interval
from modules.full_pipeline import load_data_split, WeatherBenchDatasetXarrayHealpixTemp, \
                                  train_model_2steps, create_iterative_predictions_healpix_temp, \
                                  compute_errors, plot_climatology, WeatherBenchDatasetXarrayHealpixTempMultiple
from modules.plotting import plot_general_skills, plot_benchmark, plot_skillmaps, plot_benchmark_simple,\
                             plot_intervalmap   
from modules.data import hp_to_equiangular
from modules.mail import send_info_mail

from swag import data, models, utils, losses
from swag.posteriors import SWAG


import warnings
warnings.filterwarnings("ignore")

def generate_plots_eval(prediction_ds, reference_rmses, obs, resolution, lead_time, max_lead_time, len_sqce, 
                        metrics_path, figures_path, description, rmse_spherical=None, general_skills=True, benchmark_simple=True,
                        skillmaps=True):
    start_time = len_sqce * lead_time - 6
    end_time = (6-lead_time) if (6-lead_time) > 0 else None

    # Data
    lead_times = np.arange(lead_time, max_lead_time+lead_time, lead_time)

    t = time.time()
    corr_map, rbias_map, rsd_map, rmse_map, obs_rmse, rmse_map_norm = compute_errors(prediction_ds, obs)
    print(time.time() - t)
    
    if rmse_spherical is None:
        rmse_spherical = xr.load_dataset(metrics_path + 'rmse_' + description + '.nc')
    rbias_spherical = rbias_map.mean('node').compute()
    rsd_spherical = rsd_map.mean('node').compute()
    corr_spherical = corr_map.mean('node').compute()

    if benchmark_simple:
        plot_benchmark_simple(rmse_spherical, reference_rmses, description, lead_times, 
                input_dir=metrics_path, output_dir=figures_path, title=False)
    
    if general_skills:
        plot_general_skills(rmse_map_norm, corr_map, rbias_map, rsd_map, description, lead_times, 
                        output_dir=figures_path, title=False)

    if skillmaps:
        plot_skillmaps(rmse_map_norm, rsd_map, rbias_map, corr_map, description, lead_times, resolution, 
                    output_dir=figures_path)


def hovmoller_diagram(prediction_ds, obs, lead_idx, resolution, figures_path, description):
    monthly_mean = prediction_ds.groupby('time.month').mean().compute()
    monthly_mean_obs = obs.groupby('time.month').mean().compute()

    # Computations
    monthly_mean.isel(lead_time=lead_idx)
    monthly_mean_eq = []
    for month in range(12):
        monthly_mean_eq.append(hp_to_equiangular(monthly_mean.isel(lead_time=lead_idx, month=month), 
                                        resolution))
    monthly_mean_eq = xr.concat(monthly_mean_eq, pd.Index(np.arange(1, 13, 1), name='month'))
    monthly_lat_eq = monthly_mean_eq.mean('lon')

    monthly_mean_obs.isel(lead_time=lead_idx)
    monthly_mean_eq_obs = []
    for month in range(12):
        monthly_mean_eq_obs.append(hp_to_equiangular(monthly_mean_obs.isel(lead_time=lead_idx, month=month), 
                                        resolution))
    monthly_mean_eq_obs = xr.concat(monthly_mean_eq_obs, pd.Index(np.arange(1, 13, 1), name='month'))
    monthly_lat_eq_obs = monthly_mean_eq_obs.mean('lon')

    pred_z = np.rot90(monthly_lat_eq.z.values, 3)
    pred_t = np.rot90(monthly_lat_eq.t.values, 3)
    obs_z = np.rot90(monthly_lat_eq_obs.z.values, 3)
    obs_t = np.rot90(monthly_lat_eq_obs.t.values, 3)

    diff_z = pred_z / obs_z
    diff_t = pred_t / obs_t

    # Labels and limits
    ticks = np.linspace(0, 31, 7).astype(int)
    lat_labels = np.linspace(-90, 90, 7).astype(int)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    vmin_z = min(np.min(monthly_lat_eq.z).values.flatten()[0], np.min(monthly_lat_eq_obs.z).values.flatten()[0])
    vmax_z = max(np.max(monthly_lat_eq.z).values.flatten()[0], np.max(monthly_lat_eq_obs.z).values.flatten()[0])
    vmin_t = min(np.min(monthly_lat_eq.t).values.flatten()[0], np.min(monthly_lat_eq_obs.t).values.flatten()[0])
    vmax_t = max(np.max(monthly_lat_eq.t).values.flatten()[0], np.max(monthly_lat_eq_obs.t).values.flatten()[0])

    delta = min((np.min(diff_z)-1), (1-np.max(diff_z)), (np.min(diff_t)-1), (1-np.max(diff_t)))

    vmin_sd = 1 - delta
    vmax_sd = 1 + delta

    predictions_vals = {'pred_z': pred_z, 'pred_t': pred_t,'obs_z': obs_z,'obs_t': obs_t}
    val_limits = {'vmin_z':vmin_z, 'vmax_z':vmax_z, 'vmin_t':vmin_t, \
                'vmax_t':vmax_t, 'vmin_sd':vmin_sd, 'vmax_sd':vmax_sd}
    
     
    figname = figures_path + description + '_hovmoller'
    plot_climatology(figname, predictions_vals, val_limits, ticks, lat_labels, month_labels)
    
    
def crpss(obs, forecast, dim, gaussian=True, **metric_kwargs):
    dim_for_gaussian = dim.copy()
    dim_for_gaussian.remove("member")   
    
    if gaussian: 
        mu = obs.mean(dim_for_gaussian)
        sig = obs.std(dim_for_gaussian)
        ref_skill = crps_gaussian(obs, mu, sig, dim=dim_for_gaussian)
    else:
        cdf_or_dist = metric_kwargs.pop("cdf_or_dist", norm)
        xmin = metric_kwargs.pop("xmin", None)
        xmax = metric_kwargs.pop("xmax", None)
        tol = metric_kwargs.pop("tol", 1e-6)
        ref_skill = crps_quadrature(
            forecast,
            cdf_or_dist,
            xmin=xmin,
            xmax=xmax,
            tol=tol,
            dim=dim_for_gaussian,
            **metric_kwargs,
        )
    
    forecast_skill = crps_ensemble(obs, forecast, dim=dim_for_gaussian)
    skill_score = 1 - forecast_skill / ref_skill

    return skill_score


def check_interval(prediction_ds, observations):
    min_pred = prediction_ds.min(axis=0)
    max_pred = prediction_ds.max(axis=0)
    
    interval = ((observations >= min_pred) & (observations <= max_pred))
    interval_mean = interval.mean(dim=['time', 'node']).compute()

    interval_z = interval_mean.z.values
    interval_t = interval_mean.t.values
    
    return interval_z, interval_t, interval


def generate_predictions_ensemble(config_file, nb_models=5, ensembling=False, swag=False, scale_swag=1.0, 
                                  no_cov_mat=False, max_models_swag=20, multiple_swag_realizations=False, 
                                  nb_realizations=1, aggregate_realizations=False, plot_realizations=False, 
                                  last_epoch_only=True, load_if_exists=False, load_metrics=False, probabilistic=True, 
                                  full_plots=True, hovmoller=True, file_prefix=None):
    print('Reading confing file and setting up folders...')

    pid = os.getpid()
    process = psutil.Process(pid)
    
    # load config
    with open("../configs/" + config_file) as json_data_file:
        cfg = json.load(json_data_file)

    # define paths
    datadir = cfg['directories']['datadir']
    input_dir = datadir + cfg['directories']['input_dir']
    model_save_path = datadir + cfg['directories']['model_save_path']
    pred_save_path = datadir + cfg['directories']['pred_save_path']
    metrics_path = datadir + cfg['directories']['metrics_path']

    if not os.path.isdir(pred_save_path):
        os.mkdir(pred_save_path)

    # define constants
    chunk_size = cfg['training_constants']['chunk_size']

    train_years = (cfg['training_constants']['train_years'][0], cfg['training_constants']['train_years'][1])
    val_years = (cfg['training_constants']['val_years'][0], cfg['training_constants']['val_years'][1])
    test_years = (cfg['training_constants']['test_years'][0], cfg['training_constants']['test_years'][1])

    # training parameters
    nodes = cfg['training_constants']['nodes']
    max_lead_time = cfg['training_constants']['max_lead_time']
    nb_timesteps = cfg['training_constants']['nb_timesteps']
    epochs = cfg['training_constants']['nb_epochs']
    learning_rate = cfg['training_constants']['learning_rate']
    batch_size = cfg['training_constants']['batch_size']

    # model parameters
    len_sqce = cfg['model_parameters']['len_sqce']
    delta_t = cfg['model_parameters']['delta_t']
    in_features = cfg['model_parameters']['in_features']
    out_features = cfg['model_parameters']['out_features']
    num_steps_ahead = cfg['model_parameters']['num_steps_ahead']
    architecture_name = cfg['model_parameters']['architecture_name']
    model = cfg['model_parameters']['model']
    resolution = cfg['model_parameters']['resolution']

    description = "all_const_len{}_delta_{}_architecture_".format(len_sqce, delta_t) + architecture_name
    model_filename = model_save_path + description + ".h5"

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    gpu = [0]
    num_workers = 10
    pin_memory = True

    obs = xr.open_mfdataset(pred_save_path + 'observations_nearest.nc', combine='by_coords', chunks={'time':chunk_size})
    rmses_weyn = xr.open_dataset(datadir + 'metrics/rmses_weyn.nc')

    # get training, validation and test data
    ds_train, ds_valid, ds_test = load_data_split(input_dir, train_years, val_years, test_years, chunk_size)

    constants = xr.open_dataset(f'{input_dir}constants/constants_5.625deg_standardized.nc')

    orog = constants['orog']
    lsm = constants['lsm']
    lats = constants['lat2d']
    slt = constants['slt']

    num_constants = len([orog, lats, lsm, slt])

    train_mean_ = xr.open_mfdataset(f'{input_dir}mean_train_features_dynamic.nc')
    train_std_ = xr.open_mfdataset(f'{input_dir}std_train_features_dynamic.nc')

    # generate train dataloader
    if swag and not ensembling:
        training_ds = WeatherBenchDatasetXarrayHealpixTempMultiple(ds=ds_train, out_features=out_features, delta_t=delta_t,
                                                       len_sqce_input=len_sqce, len_sqce_output=num_steps_ahead, max_lead_time=max_lead_time,
                                                       years=train_years, nodes=nodes, nb_timesteps=nb_timesteps,
                                                       mean=train_mean_, std=train_std_)

    # generate test dataloader
    testing_ds = WeatherBenchDatasetXarrayHealpixTemp(ds=ds_test, out_features=out_features,
                                                    len_sqce=len_sqce, delta_t=delta_t, years=test_years, 
                                                    nodes=nodes, nb_timesteps=nb_timesteps, 
                                                    mean=train_mean_, std=train_std_, 
                                                    max_lead_time=max_lead_time)

    

    
    constants_tensor = torch.tensor(xr.merge([orog, lats, lsm, slt], compat='override').to_array().values, \
                                    dtype=torch.float)
    # standardize
    constants_tensor = (constants_tensor - torch.mean(constants_tensor, dim=1).view(-1, 1).expand(4, 3072)) / \
                       torch.std(constants_tensor, dim=1).view(-1, 1).expand(4, 3072)
    
    constants_expanded = constants_tensor.transpose(1, 0).expand(batch_size, nodes, num_constants)
    
    crps_results = []

    epochs_range = [epochs-1] if last_epoch_only else [ep for ep in range(epochs)]
    
    for ep in epochs_range:
        print(f"########Epoch {ep}########")
        models_predictions = []

        description_epoch = description
        if file_prefix:
            description_epoch += f'_{file_prefix}'
        
        if ensembling:
            description_epoch += '_ensemble_epoch_{}'.format(ep)
        else:
            description_epoch += '_epoch_{}'.format(ep)

        if swag:
            description_epoch += '_swag_scale{}'.format(str(scale_swag).replace('.', ''))

        pred_filename = pred_save_path +  description_epoch

        if ensembling or multiple_swag_realizations:
            pred_median_filename = pred_filename + "_median.nc"
            pred_mean_filename = pred_filename + "_mean.nc"

            if aggregate_realizations:
                pred_median_filename = pred_filename + "_aggregate_median.nc"
                pred_mean_filename = pred_filename + "_aggregate_mean.nc"
            
        rmse_filename = datadir + 'metrics/rmse_' + description_epoch + '.nc'
        if ensembling or multiple_swag_realizations:
            rmse_median_filename = rmse_filename[:-3] + '_median.nc'
            rmse_mean_filename = rmse_filename[:-3] + '_mean.nc'

            if aggregate_realizations:
                rmse_median_filename = rmse_filename[:-3] + '_aggregate_median.nc'
                rmse_mean_filename = rmse_filename[:-3] + '_aggregate_mean.nc'
        
        if multiple_swag_realizations:
            rmse_realizations_filename = []
        
        if probabilistic:
            crps_filename = datadir + 'metrics/crps_' + description_epoch + '.nc'
            interval_filename = datadir + 'metrics/interval_' + description_epoch + '.nc'

            if aggregate_realizations:
                crps_filename = datadir + 'metrics/crps_' + description_epoch + '_aggregate.nc'
                interval_filename = datadir + 'metrics/interval_' + description_epoch + '_aggregate.nc'

        figures_path = '../data/healpix/figures/' + description_epoch + '/'

        if aggregate_realizations:
            figures_path = '../data/healpix/figures/' + description_epoch + '_aggregate/'
        
        if not os.path.isdir(figures_path):
            os.mkdir(figures_path)

        if not os.path.isdir(pred_save_path):
                os.mkdir(pred_save_path)

        for i in range(nb_models):
            print("\nModel", i+1)

            description_model_epoch = description
            if file_prefix:
                description_model_epoch += f'_{file_prefix}'
            if ensembling:
                description_model_epoch += '_ensemble_model{}_epoch{}'.format(i+1, ep)
            else:
                description_model_epoch += '_epoch{}'.format(ep)
            if swag:
                description_model_epoch += '_swag'

            model_name = model_save_path + description_model_epoch + '.h5'
            pred_model_filename = pred_save_path +  description_model_epoch + ".nc"
            if swag:
                pred_model_filename = pred_model_filename[:-3] + \
                                        '_scale{}.nc'.format(str(scale_swag).replace('.', ''))
            
            if ensembling:
                    model_path = '../data/healpix/models/'
                    years_file = description_model_epoch.replace("_epoch{}".format(ep), "") + ".json"
                    if swag:
                        years_file = years_file[:-10] + ".json"
                    with open(model_path + years_file) as json_data_file:
                        years = json.load(json_data_file)
                    
                    train_years = years['train_years']
                    val_years = years['val_years']
                    ds_train, _, _ = load_data_split(input_dir, train_years, val_years, test_years, chunk_size, random_split=ensembling)
                    training_ds = WeatherBenchDatasetXarrayHealpixTempMultiple(ds=ds_train, out_features=out_features, delta_t=delta_t,
                                                            len_sqce_input=len_sqce, len_sqce_output=num_steps_ahead, max_lead_time=max_lead_time,
                                                            years=train_years, nodes=nodes, nb_timesteps=nb_timesteps,
                                                            mean=train_mean_, std=train_std_)  

            if aggregate_realizations:
                realizations_predictions = []

            for j in range(nb_realizations):
                torch.manual_seed(i+j)

                if multiple_swag_realizations and nb_realizations > 1:
                    print("\nRealization", j+1)
                
                if ensembling or multiple_swag_realizations:
                    print("Current memory use :", process.memory_percent())
                    if process.memory_percent() > 35:
                        return
                
                
                if multiple_swag_realizations:
                    pred_model_filename = pred_model_filename[:-3] + '_realization{}.nc'.format(j+1)
                    rmse_realization_filename = datadir + 'metrics/rmse_' + description_model_epoch + '_realization{}.nc'.format(j+1)
                    rmse_realizations_filename.append(rmse_realization_filename)
                
                if not os.path.isfile(model_name):
                    print(model_name)
                    continue

                if load_if_exists and os.path.isfile(pred_model_filename):
                    print("\tLoading existing predictions")
                    prediction_ds = xr.open_dataset(pred_model_filename).chunk('auto')
                    if aggregate_realizations:
                        realizations_predictions.append(prediction_ds)
                    else:
                        models_predictions.append(prediction_ds)
                    continue           

                # load model
                modelClass = getattr(modelArchitectures, model)
        
                if swag:
                    model_epoch = SWAG(modelClass, no_cov_mat=no_cov_mat, max_num_models=max_models_swag, N=nodes, 
                                    in_channels=in_features * len_sqce, out_channels=out_features, kernel_size=3)
                else:
                    model_epoch = modelClass(N=nodes, in_channels=in_features * len_sqce,
                                    out_channels=out_features, kernel_size=3)

                state_to_load = torch.load(model_name)
                
                if swag:
                    model_epoch.load_state_dict(state_to_load, strict=False)
                else:
                    own_state = model_epoch.state_dict()
                    for name, param in state_to_load.items():
                        if name not in own_state:
                            if '.'.join(name.split('.')[1:]) not in own_state:
                            #own_state[name] = param
                                print(name)
                            else:
                                n = name
                                name = '.'.join(n.split('.')[1:])
                                
                            #continue
                        if isinstance(param, Parameter):
                            # backwards compatibility for serialized parameters
                            param = param.data
                        own_state[name].copy_(param)
                    
                model_epoch, device = init_device(model_epoch, gpu=gpu)
                
                if swag:
                    with torch.no_grad():
                        model_epoch.sample(scale_swag, cov=(not no_cov_mat))
                    utils.bn_update(training_ds, model_epoch, batch_size, constants_expanded, device)
                
                # compute predictions
                predictions, lead_times, times, nodes, out_lat, out_lon = \
                create_iterative_predictions_healpix_temp(model_epoch, device, testing_ds, constants_tensor.transpose(1,0))
                
                # save predictions
                das = []
                lev_idx = 0
                for var in ['z', 't']:       
                    das.append(xr.DataArray(
                        predictions[:, :, :, lev_idx],
                        dims=['lead_time', 'time', 'node'],
                        coords={'lead_time': lead_times, 'time': times[:predictions.shape[1]], 'node': np.arange(nodes)},
                        name=var
                    ))
                    lev_idx += 1

                prediction_ds = xr.merge(das)
                prediction_ds = prediction_ds.assign_coords({'lat': out_lat, 'lon': out_lon})
                prediction_ds = prediction_ds.chunk('auto')

                print("\nSaving to: ", pred_model_filename)

                # save individual model predictions
                try:
                    prediction_ds.to_netcdf(pred_model_filename)
                except PermissionError:
                    print("Can't save predictions, permission denied")
                
                if aggregate_realizations:
                    realizations_predictions.append(prediction_ds)
                else:
                    models_predictions.append(prediction_ds)

                if ensembling or multiple_swag_realizations:
                    del prediction_ds
            
            # Take median of realizations per model
            if aggregate_realizations:
                out_lat = realizations_predictions[0]['lat']
                out_lon = realizations_predictions[0]['lon']
                realizations_predictions = [pred.drop('lat').drop('lon') for pred in realizations_predictions]
                prediction_ds = xr.concat(realizations_predictions, dim="member")
                predictions_median = [prediction_ds['z'].median(axis=0), prediction_ds['t'].median(axis=0)]
                prediction_ds = xr.merge(predictions_median)
                prediction_ds = prediction_ds.assign_coords({'lat': out_lat, 'lon': out_lon}).chunk(realizations_predictions[0].chunks)
                models_predictions.append(prediction_ds)

                pred_model_filename_agg = pred_model_filename[:-3] + "_aggregate.nc"
                prediction_ds.to_netcdf(pred_model_filename_agg)

                del prediction_ds
                del realizations_predictions
            
        if ensembling or multiple_swag_realizations:
            print("Current memory use :", process.memory_percent())
            if process.memory_percent() > 35:
                return
        
        if load_if_exists:
            out_lat = models_predictions[0]['lat']
            out_lon = models_predictions[0]['lon']

        # concatenate all predictions from different models
        if ensembling or multiple_swag_realizations:
            models_predictions = [pred.drop('lat').drop('lon') for pred in models_predictions]
            prediction_ds = xr.concat(models_predictions, dim="member")
        
        # load observations
        obs = xr.open_mfdataset(pred_save_path + 'observations_nearest.nc', combine='by_coords')
        # obs = obs.isel(time=slice(6,prediction_ds.time.shape[0]+6))
        common_time = list(set(prediction_ds.time.values).intersection(obs.time.values))
        common_time.sort()
        common_lead_time = list(set(prediction_ds.lead_time.values).intersection(obs.lead_time.values))
        common_lead_time.sort()
        prediction_ds = prediction_ds.sel(dict(time=common_time, lead_time=common_lead_time))
        obs = obs.sel(dict(time=common_time, lead_time=common_lead_time))
        obs = obs.chunk(models_predictions[-1].chunks)
        
        if ensembling or multiple_swag_realizations:
            if probabilistic:
                # compute crps
                if load_metrics and os.path.isfile(crps_filename):
                    crps_epoch = xr.open_dataset(crps_filename)
                else:
                    prediction_ds = prediction_ds
                    dims = list(prediction_ds.dims).copy()
                    dims.remove('member')
                    dims.remove('lead_time')
                    crps_epoch = crps_ensemble(obs, prediction_ds.chunk({'member': -1}), dim=dims).compute()
                    crps_epoch.to_netcdf(crps_filename)
                #skill_score = crpss(obs, prediction_ds, dims=list(prediction_ds.dims))
                crps_results.append(list(zip(crps_epoch.z.values, crps_epoch.t.values)))
                print('\nCRPS:')
                print('\tZ500 : {:.3f}'.format(crps_results[-1][0][0]))
                print('\tT850 : {:.3f}'.format(crps_results[-1][0][1]))

                plot_crps(crps_epoch, lead_time=6, max_lead_time=max_lead_time)

                # Compute percentage of observations in ensemble interval
                interval_z, interval_t, interval = check_interval(prediction_ds, obs)
                print('Percentage of observations in the ensemble interval')
                print('\tZ500 : {:.3%}'.format(interval_z[0]))
                print('\tT850 : {:.3%}'.format(interval_t[0]))

                plot_interval(interval_z, interval_t, lead_time=6, max_lead_time=max_lead_time)
                
                interval = interval.drop('lat').drop('lon')
                interval = interval.astype('uint8').mean(['time']).compute()
                
                lead_time = 6
                lead_times = np.arange(lead_time, max_lead_time + lead_time, lead_time)
                plot_intervalmap(interval, description_epoch + '_median', lead_times, resolution, figures_path)

            models_predictions = [pred.assign_coords({'lat': out_lat, 'lon': out_lon}) for pred in models_predictions]

            if load_metrics and os.path.isfile(pred_median_filename):
                prediction_median_ds = xr.open_dataset(pred_median_filename).chunk(models_predictions[0].chunks)
            else:
                t_median_start = time.time()
                # median over values of ensemble
                predictions_median = [prediction_ds['z'].median(axis=0), prediction_ds['t'].median(axis=0)]
                prediction_median_ds = xr.merge(predictions_median)
                prediction_median_ds = prediction_median_ds.assign_coords({'lat': out_lat, 'lon': out_lon}).chunk(models_predictions[0].chunks)
                
                # save final median predictions
                prediction_median_ds.to_netcdf(pred_median_filename)
                t_median_end = time.time()
                print("Median predictions : {t:2f}".format(t=t_median_end-t_median_start))

            if load_metrics and os.path.isfile(pred_mean_filename):
                prediction_mean_ds = xr.open_dataset(pred_mean_filename).chunk(models_predictions[0].chunks)
            else:
                t_mean_start = time.time()
                # mean over values of ensemble
                predictions_mean = [prediction_ds['z'].mean(axis=0), prediction_ds['t'].mean(axis=0)]
                prediction_mean_ds = xr.merge(predictions_mean)
                prediction_mean_ds = prediction_mean_ds.assign_coords({'lat': out_lat, 'lon': out_lon}).chunk(models_predictions[0].chunks)
                
                # save final mean predictions
                prediction_median_ds.to_netcdf(pred_mean_filename)
                t_mean_end = time.time()
                print("Mean predictions : {t:2f}".format(t=t_mean_end-t_mean_start))

            # compute RMSE
            reference_rmses = rmses_weyn.rename({'z500':'z', 't850':'t'}).sel(lead_time=common_lead_time)
            
            if load_metrics and os.path.isfile(rmse_median_filename):
                rmse_median = xr.open_dataset(rmse_median_filename)
            else:
                t_median_start = time.time()
                ## RMSE for median of predictions
                rmse_median = compute_rmse_healpix(prediction_median_ds, obs).load()
                rmse_median.to_netcdf(rmse_median_filename)
                t_median_end = time.time()
                print("RMSE median : {t:2f}".format(t=t_median_end-t_median_start))
            
            if load_metrics and os.path.isfile(rmse_mean_filename):
                rmse_mean = xr.open_dataset(rmse_mean_filename)
            else:
                t_mean_start = time.time()
                ## RMSE for mean of predictions
                rmse_mean = compute_rmse_healpix(prediction_mean_ds, obs).load()
                rmse_mean.to_netcdf(rmse_mean_filename)
                t_mean_end = time.time()
                print("RMSE mean : {t:2f}".format(t=t_mean_end-t_mean_start))

            # plot RMSE
            print('\nRMSE median')
            print('\tZ500 - 0:', rmse_median.z.values[0])
            print('\tT850 - 0:', rmse_median.t.values[0])

            # plot RMSE
            print('RMSE mean')
            print('\tZ500 - 0:', rmse_mean.z.values[0])
            print('\tT850 - 0:', rmse_mean.t.values[0])

            ## Computing the RMSE of individual realizations of SWAG
            if plot_realizations:
                t_realizations_start = time.time()
                rmses_realizations = []
                for idx, realization in enumerate(models_predictions):
                    rmse_realization = compute_rmse_healpix(realization, obs)
                    rmse_realization.to_netcdf(rmse_realizations_filename[idx])
                    rmses_realizations.append(rmse_realization)
                t_realizations_end = time.time()
                print("\nRMSE realizations : {t:2f}".format(t=t_realizations_end-t_realizations_start))

            if plot_realizations:
                # Plot the realizations and their median
                plot_rmses_realizations(rmse_median, rmses_realizations, lead_time=6, max_lead_time=max_lead_time, 
                                        title=f'Realizations and median comparison for scale {str(scale_swag)}', rmse_mean=rmse_mean)
                
                lead_time = 6
                generate_plots_eval(prediction_median_ds, reference_rmses, obs, resolution, lead_time, max_lead_time, len_sqce, metrics_path, 
                                    figures_path, description_epoch + "_median", rmse_spherical=rmse_median, general_skills=True, 
                                    benchmark_simple=False, skillmaps=False)
            if ensembling:
                plot_rmses(rmse_median, reference_rmses, lead_time=6, max_lead_time=max_lead_time, 
                            title=f'RMSE of Median of Ensemble for {nb_models} models')   

                # plot_rmses(rmse_mean, reference_rmses, lead_time=6, max_lead_time=max_lead_time,
                #             title=f'RMSE of Mean of Ensemble for {nb_models} models')
        else:
            # compute RMSE
            reference_rmses = rmses_weyn.rename({'z500':'z', 't850':'t'}).sel(lead_time=common_lead_time)
            # RMSE of predictions
            rmse = compute_rmse_healpix(prediction_ds, obs).load()
            rmse.to_netcdf(rmse_filename)
            
            # plot RMSE
            print('RMSE')
            print('\tZ500 - 0:', rmse.z.values[0])
            print('\tT850 - 0:', rmse.t.values[0])

            plot_rmses(rmse, reference_rmses, lead_time=6, max_lead_time=max_lead_time)

        torch.cuda.empty_cache()
    
    plot_params = {
        'resolution': resolution,
        'max_lead_time': max_lead_time,
        'len_sqce': len_sqce,
        'metrics_path': metrics_path,
        'figures_path': figures_path, 
        'description_epoch' : description_epoch
    }
    
    if full_plots:
        lead_time = 6
        if ensembling or multiple_swag_realizations:
            print("Median")
            generate_plots_eval(prediction_median_ds, reference_rmses, obs, resolution, lead_time, max_lead_time, len_sqce, metrics_path, 
                                figures_path, description_epoch + '_median')
            # print("Mean")
            # generate_plots_eval(prediction_mean_ds, reference_rmses, obs, resolution, lead_time, max_lead_time, len_sqce, metrics_path, 
            #                     figures_path, description_epoch + '_mean')
        else:
            generate_plots_eval(prediction_ds, reference_rmses, obs, resolution, lead_time, max_lead_time, len_sqce, metrics_path, 
                                figures_path, description_epoch)
    
    if hovmoller:
        lead_idx = len(lead_times) - 1
        if ensembling or multiple_swag_realizations:
            print("Median")
            hovmoller_diagram(prediction_median_ds, obs, lead_idx, resolution, figures_path, description_epoch + '_median')

            # print("Mean")
            # hovmoller_diagram(prediction_mean_ds, obs, lead_idx, resolution, figures_path, description_epoch + '_mean')
        else:
            hovmoller_diagram(prediction_ds, obs, lead_idx, resolution, figures_path, description_epoch)
    
    if ensembling or multiple_swag_realizations:
        return crps_results, rmse_median, rmse_mean, prediction_ds, prediction_mean_ds, prediction_median_ds, obs, \
                plot_params
    else:
        return rmse, prediction_ds, obs, plot_params


