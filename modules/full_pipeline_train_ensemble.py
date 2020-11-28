import os
import json
import time
import pickle
import random
import argparse

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))

import modules.architectures as modelArchitectures

from modules.plotting import plot_rmses
from modules.utils import init_device
from modules.test import compute_rmse_healpix
from modules.full_pipeline import load_data_split, WeatherBenchDatasetXarrayHealpixTemp, \
    train_model_2steps, create_iterative_predictions_healpix_temp, \
    compute_errors, plot_climatology, WeatherBenchDatasetXarrayHealpixTempMultiple

from swag import data, models, utils, losses
from swag.posteriors import SWAG

import warnings
warnings.filterwarnings("ignore")

import faulthandler
faulthandler.enable()


# Adapted : added model_nb argument to capture the model number of the ensemble, and the train split
# as the training set will be split randomly
def main_ensemble(config_file, model_nb, plot_weight_variations=False, ensembling=True, train_split=0.8,
                  swag=False, swa_start=3, no_cov_mat=False, max_num_models_swag=20, save_only_last=True, 
                  load_model=False, model_name_epochs=None, file_prefix=None):
    def update_w(w):
        """
        Update array of weights for the loss function
        :param w: array of weights from earlier step [0] to latest one [-1]
        :return: array of weights modified
        """
        for i in range(1, len(w)):
            len_w = len(w)
            w[len_w - i] += w[len_w - i -1]*0.4
            w[len_w - i - 1] *= 0.8
        w = np.array(w)/sum(w)

        return w

    def train_model_multiple_steps(model, weights_loss, criterion, optimizer, device, training_ds,
                                   len_output, constants, batch_size, epochs, validation_ds, model_filename, 
                                   swag, swag_model=None, no_cov_mat=False):
        
        
        # Initialize parameters and storage of results
        t1 = time.time()
        train_losses = []
        val_losses = []
        swa_val_losses = []
        
        n_samples = training_ds.n_samples
        n_samples_val = validation_ds.n_samples
        num_nodes = training_ds.nodes
        num_constants = constants.shape[1]
        out_features = training_ds.out_features
        num_in_features = training_ds.in_features + num_constants

        # Expand constants to match batch size
        constants_expanded = constants.expand(batch_size, num_nodes, num_constants)
        constants1 = constants_expanded.to(device)
        idxs_val = validation_ds.idxs

        required_output = np.max(np.where(weights_loss > 0)) + 1 #produce only predictions of time-steps that have a positive contribution to the loss, otherwise, if the loss is 0 and they are not required for the prediction of future time-steps, avoid computation to save time and memory

        # save weight modifications along training phase
        weight_variations = [(weights_loss, 0, 0)]
        count_upd = 0
        
        # save loss for different time-ahead predictions to asses effect of weight variations
        train_loss_steps = {}
        for step_ahead in range(len_output):
            train_loss_steps['t{}'.format(step_ahead)] = []

        test_loss_steps = {}
        for step_ahead in range(len_output):
            test_loss_steps['t{}'.format(step_ahead)] = []

        threshold = 1e-4

        print('Time initial steps: {:.3f}s'.format(time.time() - t1))

        # iterate along epochs
        for epoch in range(epochs):

            print('\rEpoch : {}'.format(epoch), end="")

            time1 = time.time()

            val_loss = 0
            train_loss = 0

            model.train()

            random.shuffle(training_ds.idxs)
            idxs = training_ds.idxs

            batch_idx = 0
            train_loss_it = []
            times_it = []
            t0 = time.time()
            
            # iterate along batches 
            for i in range(0, n_samples - batch_size, batch_size):
                #tbatch0 = time.time()
                i_next = min(i + batch_size, n_samples)

                # addapt constants size if necessary
                if len(idxs[i:i_next]) < batch_size:
                    constants_expanded = constants.expand(len(idxs[i:i_next]), num_nodes, num_constants)
                    constants1 = constants_expanded.to(device)

                batch, labels = training_ds[idxs[i:i_next]]

                # Transfer to GPU
                batch_size = batch[0].shape[0] // 2

                batch1 = torch.cat((batch[0][:batch_size, :, :], \
                                    constants_expanded, batch[0][batch_size:, :, :], constants_expanded), dim=2).to(device)

                #  generate predictions multiple steps ahead sequentially
                output = model(batch1)
                label1 = labels[0].to(device)
                l0 = criterion(output, label1[batch_size:, :, :out_features])
                loss_ahead = weights_loss[0] * l0
                train_loss_steps['t0'].append(l0.item())

                #tbatch1 = time.time()
                for step_ahead in range(1, required_output):
                    # input t-2
                    inp_t2 = batch1[:, :, num_in_features:]

                    #  toa at t-1
                    toa_delta = labels[step_ahead][:batch_size, :, -1].view(-1, num_nodes, 1).to(device)
                    batch1 = torch.cat((inp_t2, output, toa_delta, constants1), dim=2)

                    output = model(batch1)
                    label1 = labels[step_ahead].to(device)
                    
                    # evaluate loss 
                    l0 = criterion(output, label1[batch_size:, :, :out_features])
                    loss_ahead += weights_loss[step_ahead] * l0
                    train_loss_steps['t{}'.format(step_ahead)].append(l0.item())

                #tbatch2 = time.time()
                optimizer.zero_grad()
                loss_ahead.backward()
                optimizer.step()

                #tbatch3 = time.time()

                train_loss += loss_ahead.item() * batch_size
                train_loss_it.append(train_loss / (batch_size * (batch_idx + 1)))
                
                # update weights 
                if len(train_loss_it) > 5:
                    # allow weight updates if loss does not change after a certain number of epochs (count_upd)
                    if (np.std(train_loss_it[-10:]) < threshold) and count_upd > 2e2:
                        weights_loss = update_w(weights_loss)
                        required_output = np.max(np.where(weights_loss > 0)) + 1 # update based on weights 
                        count_upd = 0
                        # print('New weights ', weights_loss, ' Epoch {} Iter {}'.format(epoch, i))
                        weight_variations.append((weights_loss, epoch, len(train_loss_steps['t0'])))
                        threshold /= 10
                    else:
                        count_upd += 1

                #tbatch4 = time.time()
                times_it.append(time.time() - t0)
                t0 = time.time()

                if batch_idx % 50 == 0:
                    print('\rBatch idx: {}; Loss: {:.3f} - Other {:.5f} - {}' \
                          .format(batch_idx, train_loss / (batch_size * (batch_idx + 1)), np.std(train_loss_it[-10:]),
                                  count_upd),
                          end="")
                batch_idx += 1


            train_loss = train_loss / n_samples
            train_losses.append(train_loss)

            # SWAG model training
            if swag and swag_model:
                swag_model.collect_model(model)
                swag_model.sample(0.0)
                utils.bn_update(training_ds, swag_model, batch_size, constants_expanded, device)

            model.eval()

            constants1 = constants_expanded.to(device)
            with torch.set_grad_enabled(False):
                index = 0

                for i in range(0, n_samples_val - batch_size, batch_size):
                    i_next = min(i + batch_size, n_samples_val)

                    if len(idxs_val[i:i_next]) < batch_size:
                        constants_expanded = constants.expand(len(idxs_val[i:i_next]), num_nodes, num_constants)
                        constants1 = constants_expanded.to(device)

                    batch, labels = training_ds[idxs[i:i_next]]

                    # Transfer to GPU
                    batch_size = batch[0].shape[0] // 2
                    batch1 = torch.cat((batch[0][:batch_size, :, :], \
                                        constants_expanded, batch[0][batch_size:, :, :], constants_expanded), dim=2).to(
                        device)

                    #  generate predictions multiple steps ahead sequentially
                    output = model(batch1)
                    label1 = labels[0].to(device)
                    l0 = criterion(output, label1[batch_size:, :, :out_features]).item()
                    loss_ahead = weights_loss[0] * l0
                    test_loss_steps['t0'].append(l0)

                    for step_ahead in range(1, required_output):
                        # input t-2
                        inp_t2 = batch1[:, :, num_in_features:]

                        #  toa at t-1
                        toa_delta = labels[step_ahead][:batch_size, :, -1].view(-1, num_nodes, 1).to(device)
                        batch1 = torch.cat((inp_t2, output, toa_delta, constants1), dim=2)

                        output = model(batch1)
                        label1 = labels[step_ahead].to(device)
                        l0 = criterion(output, label1[batch_size:, :, :out_features]).item()
                        loss_ahead += weights_loss[step_ahead] * l0
                        test_loss_steps['t{}'.format(step_ahead)].append(l0)

                    val_loss += loss_ahead * batch_size
                    index = index + batch_size

            val_loss = val_loss / n_samples_val
            val_losses.append(val_loss)

            time2 = time.time()

            # Print stuff
            print('\tEpoch: {e:3d}/{n_e:3d}  - loss: {l:.3f}  - val_loss: {v_l:.5f}  - time: {t:2f}'
                  .format(e=epoch + 1, n_e=epochs, l=train_loss, v_l=val_loss, t=time2 - time1))

            # SWAG model evaluation
            if swag and swag_model:
                swa_val_loss = 0
                constants1 = constants_expanded.to(device)
                with torch.set_grad_enabled(False):
                    index = 0

                    for i in range(0, n_samples_val - batch_size, batch_size):
                        i_next = min(i + batch_size, n_samples_val)

                        if len(idxs_val[i:i_next]) < batch_size:
                            constants_expanded = constants.expand(len(idxs_val[i:i_next]), num_nodes, num_constants)
                            constants1 = constants_expanded.to(device)

                        batch, labels = training_ds[idxs[i:i_next]]

                        # Transfer to GPU
                        batch_size = batch[0].shape[0] // 2
                        batch1 = torch.cat((batch[0][:batch_size, :, :], \
                                            constants_expanded, batch[0][batch_size:, :, :], constants_expanded), dim=2).to(
                            device)

                        #  generate predictions multiple steps ahead sequentially
                        output = swag_model(batch1)
                        label1 = labels[0].to(device)
                        l0 = criterion(output, label1[batch_size:, :, :out_features]).item()
                        swa_loss_ahead = weights_loss[0] * l0
                        # test_loss_steps['t0'].append(l0)

                        for step_ahead in range(1, required_output):
                            # input t-2
                            inp_t2 = batch1[:, :, num_in_features:]

                            #  toa at t-1
                            toa_delta = labels[step_ahead][:batch_size, :, -1].view(-1, num_nodes, 1).to(device)
                            batch1 = torch.cat((inp_t2, output, toa_delta, constants1), dim=2)

                            output = swag_model(batch1)
                            label1 = labels[step_ahead].to(device)
                            l0 = criterion(output, label1[batch_size:, :, :out_features]).item()
                            swa_loss_ahead += weights_loss[step_ahead] * l0
                            #test_loss_steps['t{}'.format(step_ahead)].append(l0)

                        swa_val_loss += swa_loss_ahead * batch_size
                        index = index + batch_size

                swa_val_loss = swa_val_loss / n_samples_val
                swa_val_losses.append(val_loss)

                time3 = time.time()

                # Print stuff
                print('\tEpoch: {e:3d}/{n_e:3d}  - loss: {l:.3f}  - SWA val_loss: {v_l:.5f}  - time: {t:2f}'
                    .format(e=epoch + 1, n_e=epochs, l=train_loss, v_l=swa_val_loss, t=time3 - time2))


        return train_losses, val_losses, swa_val_losses, train_loss_it, times_it, train_loss_steps, test_loss_steps, \
               weight_variations, weights_loss, criterion, optimizer


    # Fix seed
    torch.manual_seed(model_nb)
    random.seed(model_nb)

    with open("../configs/" + config_file) as json_data_file:
        cfg = json.load(json_data_file)

    # Define paths
    datadir = cfg['directories']['datadir']
    input_dir = datadir + cfg['directories']['input_dir']
    model_save_path = datadir + cfg['directories']['model_save_path']
    pred_save_path = datadir + cfg['directories']['pred_save_path']
    metrics_path = datadir + cfg['directories']['metrics_path']

    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)

    if not os.path.isdir(pred_save_path):
        os.mkdir(pred_save_path)

    # Define constants
    chunk_size = cfg['training_constants']['chunk_size']

    train_years = (cfg['training_constants']['train_years'][0], cfg['training_constants']['train_years'][1])
    val_years = (cfg['training_constants']['val_years'][0], cfg['training_constants']['val_years'][1])
    test_years = (cfg['training_constants']['test_years'][0], cfg['training_constants']['test_years'][1])

    # Adapted : Randomly split training data
    ## Pick random training years
    if ensembling:
        start_train_year = train_years[0]
        end_train_year = val_years[1]
        year_range = [i for i in range(int(start_train_year), int(end_train_year)+1)]
       
        train_years = sorted(random.sample(year_range, int(train_split*len(year_range))))
        val_years = list(set(year_range) - set(train_years))


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

    description = "all_const_len{}_delta_{}_architecture_".format(len_sqce, delta_t) + architecture_name


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    gpu = [0]
    num_workers = 10
    pin_memory = True

    # Adapted : use random split instead of a range for the load_data_split
    # get training, validation and test data
    ds_train, ds_valid, ds_test = load_data_split(input_dir, train_years, val_years, test_years, 
                                                    chunk_size, random_split=ensembling)

    constants = xr.open_dataset(f'{input_dir}constants/constants_5.625deg_standardized.nc')

    orog = constants['orog']
    lsm = constants['lsm']
    lats = constants['lat2d']
    slt = constants['slt']

    num_constants = len([orog, lats, lsm, slt])

    train_mean_ = xr.open_mfdataset(f'{input_dir}mean_train_features_dynamic.nc')
    train_std_ = xr.open_mfdataset(f'{input_dir}std_train_features_dynamic.nc')

    # define model name

    description = "all_const_len{}_delta_{}_architecture_".format(len_sqce, delta_t) + architecture_name

    model_filename = model_save_path + description + ".h5"
    figures_path = '../data/healpix/figures/' + description + '/'

    os.makedirs(figures_path, exist_ok=True)

    # save train and val years for ensembling reproducibility
    if ensembling:
        ensembling_params = {
            'train_years': train_years,
            'val_years': val_years
        }

        filename = model_filename[:-3]
        if file_prefix:
            filename += f'_{file_prefix}'
        
        filename += '_ensemble_model{}'.format(model_nb) + '.json'

        print("Saving train/val years of ensemble to:", filename)
        
        with open(filename, 'w+', encoding='utf-8') as f:
            json.dump(ensembling_params, f, indent=4)

    # generate dataloaders
    training_ds = WeatherBenchDatasetXarrayHealpixTempMultiple(ds=ds_train, out_features=out_features, delta_t=delta_t,
                                                       len_sqce_input=len_sqce, len_sqce_output=num_steps_ahead, max_lead_time=max_lead_time,
                                                       years=train_years, nodes=nodes, nb_timesteps=nb_timesteps,
                                                       mean=train_mean_, std=train_std_)

    validation_ds = WeatherBenchDatasetXarrayHealpixTempMultiple(ds=ds_valid, out_features=out_features, delta_t=delta_t,
                                                         len_sqce_input=len_sqce, len_sqce_output=num_steps_ahead, max_lead_time=max_lead_time,
                                                         years=val_years, nodes=nodes, nb_timesteps=nb_timesteps,
                                                         mean=train_mean_, std=train_std_)

    # generate model
    print('Define model...')
    print('Model name: ', description)
    modelClass = getattr(modelArchitectures, model)
    spherical_unet = modelClass(N=nodes, in_channels=in_features * len_sqce,
                                out_channels=out_features, kernel_size=3)

    # use pretrained model to start training
    if load_model:
        model_path = '../data/healpix/models/'
        #model_name_epochs = 'all_const_len2_delta_6_architecture_loss_v0_8steps_variation0_residual_only_enc_l3_per_epoch'
        model_name = model_path + model_name_epochs

        state_to_load = torch.load(model_name)

        own_state = spherical_unet.state_dict()
        for name, param in state_to_load.items():
            if name not in own_state:
                # own_state[name] = param
                print(name)
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    spherical_unet, device = init_device(spherical_unet, gpu=gpu)

    # If SWAG is applied, initialize SWAG model and send it to GPU
    if swag:
        swag_model = SWAG(modelClass, no_cov_mat=no_cov_mat, max_num_models=max_num_models_swag, N=nodes, 
                          in_channels=in_features * len_sqce, out_channels=out_features, kernel_size=3)
        swag_model.to(device)


    constants_tensor = torch.tensor(xr.merge([orog, lats, lsm, slt], compat='override').to_array().values, \
                                    dtype=torch.float)
    # standardize
    constants_tensor = (constants_tensor - torch.mean(constants_tensor, dim=1).view(-1, 1).expand(4, 3072)) / \
                       torch.std(constants_tensor, dim=1).view(-1, 1).expand(4, 3072)

    # initialize weights
    w = cfg['model_parameters']['initial_weights']
    w = np.array(w)
    w = w / sum(w)

    if plot_weight_variations:
        # plot variation of weights
        f, ax = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
        ax = ax.flatten()
        x_vals = list(range(len(w)))
        ax[0].scatter(x_vals, w)
        ax[0].set_title('Initial weights')

        for i in range(15):
            w = update_w(w)
            ax[i + 1].scatter(x_vals, w)
            ax[i + 1].set_title('Update ' + str(i))

        plt.xlabel('Time step ahead')
        plt.ylabel('Weight')
        plt.tight_layout()
        plt.savefig(figures_path + 'weight_updates.png')
        plt.show()

    train_loss_ev = []
    val_loss_ev = []
    swa_val_loss_ev = []
    train_loss_steps_ev = []
    test_loss_steps_ev = []
    weight_variations_ev = []

    criterion = nn.MSELoss()
    optimizer = optim.Adam(spherical_unet.parameters(), lr=learning_rate, eps=1e-7, weight_decay=0, amsgrad=False)

    # train model
    for ep in range(epochs):

        print('Starting epoch {}'.format(ep + 1))

        # Adapted : variable indicating if ep is the last epoch
        last_epoch = ep == epochs-1

        spherical_unet.train()

        w = cfg['model_parameters']['initial_weights']
        w = np.array(w)
        w = w / sum(w)

        # If we apply SWAG, use a scheduler that will modify the learning rate throughout the
        # training
        if swag:
            lr = utils.schedule(ep, learning_rate, epochs, swag, swa_start, swa_lr=0.01)
            utils.adjust_learning_rate(optimizer, lr)
        
        # Check whether to train swag model or not at this epoch
        apply_swag = swag and ep >= swa_start

        # training is set to 1 epoch (and loop is performed outside) to save models and reinitialize weights 
        # THIS CAN BE EASILY IMPROVED CODE-WISE AND SIMPLIFIED! 
        train_losses, val_losses, swa_val_losses, _, _, train_loss_steps, test_loss_steps, weight_variations, \
        w, criterion, optimizer = \
            train_model_multiple_steps(spherical_unet, w, criterion, optimizer, device, training_ds, len_output=num_steps_ahead, \
                                       constants=constants_tensor.transpose(1, 0), batch_size=batch_size, epochs=1, \
                                       validation_ds=validation_ds, model_filename=model_filename, swag=apply_swag, 
                                       swag_model=swag_model if swag else None)

        train_loss_ev.append(train_losses)
        val_loss_ev.append(val_losses)
        swa_val_loss_ev.append(swa_val_losses)
        train_loss_steps_ev.append(train_loss_steps)
        test_loss_steps_ev.append(test_loss_steps)
        weight_variations_ev.append(weight_variations)

        # save model
        # Possible name formats:
        # 1. model_save_path + description + _file_prefix + _epoch + .h5
        # 2. model_save_path + description + _file_prefix + _ensemble_model_epoch + .h5
        # If SWAG model is trained, save it by adding _swag just before the extension of
        # filename
        if (save_only_last and last_epoch) or not save_only_last:
            filename = model_filename[:-3]
            if file_prefix:
                filename += f'_{file_prefix}'
            if ensembling:
                filename += '_ensemble_model{}_epoch{}'.format(model_nb, ep) + '.h5'
            else:
                filename += '_epoch{}'.format(ep) + '.h5'
            print("Saving to", filename)
            torch.save(spherical_unet.state_dict(), filename)

            if swag:
                swag_filename = filename[:-3] + '_swag.h5'
                print("Saving SWAG model to", swag_filename)
                torch.save(swag_model.state_dict(), swag_filename)

        torch.cuda.empty_cache()

    # Save results in a pickle file (because the dictionary contains list containing 
    # numpy arrays, that are non JSON serializable)
    results = {
        'config_file': config_file,
        'model_nb': model_nb,
        'ensemble': ensembling,
        'swag': swag,
        'swa_start': swa_start if swag else None,
        'max_num_models_swag': max_num_models_swag if swag else None,
        'no_cov_mat': no_cov_mat if swag else None,
        'train_years': train_years,
        'val_years': val_years,
        'train_loss_ev': train_loss_ev, 
        'val_loss_ev': val_loss_ev,
        'swa_val_loss_ev': swa_val_loss_ev, 
        'train_loss_steps_ev': train_loss_steps_ev, 
        'test_loss_steps_ev': test_loss_steps_ev, 
        'weight_variations_ev': weight_variations_ev
    }

    results_filename = filename[:-3] + '_results.pkl'

    print("Saving results and hyperparameters to:", results_filename)
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)

    return train_loss_ev, val_loss_ev, train_loss_steps_ev, test_loss_steps_ev, weight_variations_ev


def train_ensemble(config_file, nb_models=3, plot_weight_variations=False, ensembling=True, train_split=0.8, 
                   swag=False, swa_start=8, no_cov_mat=False, max_num_models_swag=20, save_only_last=True, load_model=False, 
                   model_name_epochs=None, file_prefix=None):
    train_losses = []
    val_losses = []
    train_losses_steps = []
    test_losses_steps = []
    weight_variations = []

    for model_nb in range(nb_models):
        print(f"Model {model_nb+1}:\n")
        train_loss_ev, val_loss_ev, train_loss_steps_ev, test_loss_steps_ev, weight_variations_ev = \
        main_ensemble(config_file, model_nb=model_nb+1, plot_weight_variations=plot_weight_variations, 
                      ensembling=ensembling, train_split=train_split, swag=swag, swa_start=swa_start,
                      no_cov_mat=no_cov_mat, max_num_models_swag=max_num_models_swag,save_only_last=save_only_last, 
                      load_model=load_model, model_name_epochs=model_name_epochs, file_prefix=file_prefix)
        train_losses.append(train_loss_ev)
        val_losses.append(val_loss_ev)
        train_losses_steps.append(train_loss_steps_ev)
        test_losses_steps.append(test_loss_steps_ev)
        weight_variations.append(weight_variations_ev)
    
    return train_losses, val_losses, train_losses_steps, test_losses_steps, weight_variations


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="config_residual_multiple_steps.json",
                        help='Name of config file with all the training parameters')
    parser.add_argument('--ensembling', action='store_true',
                        help='Apply deep ensembling')
    parser.add_argument('--nb_models', type=int, default=10,
                        help='Number of models to train')
    parser.add_argument('--plot_weight_variations', action='store_true',
                        help='Plot variations of weights')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Percentage of the data which is going to go for training')
    parser.add_argument('--save_only_last', action='store_true',
                        help='Save only last epoch') 
    parser.add_argument('--load_model', action='store_true', 
                        help='Load model')
    parser.add_argument('--model_name_epochs', type=str)
    parser.add_argument('--file_prefix', type=str,
                        help='Filename prefix to identify models')
    parser.add_argument('--swag', action='store_true',
                        help='Apply SWAG to model')
    parser.add_argument('--swa_start', type=int, default=8,
                        help='Epoch at which SWAG starts to be applied')
    parser.add_argument('--no_cov_mat', action='store_true',
                        help='Deactivate saving sample covariance matrix (no SWAG)')
    parser.add_argument('--max_num_models_swag', type=int, default=20,
                        help='Max number of models to save for SWAG')

    args = parser.parse_args()
    
    train_losses, val_losses, train_losses_steps, test_losses_steps, weight_variations = \
            train_ensemble(args.config_file, args.nb_models, args.plot_weight_variations, args.ensembling,
                           args.train_split, args.swag, args.swa_start, args.no_cov_mat, args.max_num_models_swag, 
                           args.save_only_last, args.load_model, args.model_name_epochs, args.file_prefix)
    