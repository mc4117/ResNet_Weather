# File which trains stacking neural networks for t850 for 5-day forecast using inputs generated using indiv_member_train_t_120.py

import argparse
# defined command line options

CLI=argparse.ArgumentParser()

import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import itertools
from tensorflow.keras.utils import to_categorical
from score import *
import re
from collections import OrderedDict

CLI.add_argument(
  "--block_no",
  type = int,
  default = 2,
)

args = CLI.parse_args()

print(args.block_no)

block_no = args.block_no

DATADIR = '/rds/general/user/mc4117/home/WeatherBench/data/'

var_dict = {
    'geopotential': ('z', [500]),
    'temperature': ('t', [850])}    

ds_list = []

for long_var, params in var_dict.items():
    if long_var == 'constants':
        ds_list.append(xr.open_mfdataset(f'{DATADIR}/{long_var}/*.nc', combine='by_coords'))
    else:
        var, levels = params
        if levels is not None:
            ds_list.append(xr.open_mfdataset(f'{DATADIR}/{long_var}/*.nc', combine='by_coords').sel(level = levels))
        else:
            ds_list.append(xr.open_mfdataset(f'{DATADIR}/{long_var}/*.nc', combine='by_coords'))

# For the data generator all variables have to be merged into a single dataset.            
# Because the first values of solar radiation are missing we exclude the first 7hrs from the dataset
ds_whole = xr.merge(ds_list).isel(time = slice(7, None))

# split dataset
ds_train = ds_whole.sel(time=slice('1979', '2016'))  
ds_test = ds_whole.sel(time=slice('2017', '2018'))
ds_valid = ds_whole.sel(time=slice('2012', '2016'))

class DataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True, 
                 mean=None, std=None, output_vars= None, bins_t = None):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            ds: Dataset containing all variables
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            lead_time: Lead time in hours
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
            bins_t: Bounds on bins in order to bin continuous weather data. If None, compute from data
        """
        
        self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time

        data = []
        level_names = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for long_var, params in var_dict.items():
            if long_var == 'constants': 
                for var in params:
                    data.append(ds[var].expand_dims(
                        {'level': generic_level, 'time': ds.time}, (1, 0)
                    ))
                    level_names.append(var)
            else:
                var, levels = params
                try:
                    data.append(ds[var].sel(level=levels))
                    level_names += [f'{var}_{level}' for level in levels]
                except ValueError:
                    data.append(ds[var].expand_dims({'level': generic_level}, 1))
                    level_names.append(var)   

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.data['level_names'] = xr.DataArray(
            level_names, dims=['level'], coords={'level': self.data.level})
        if output_vars is None:
            self.output_idxs = range(len(ds.data.level))
        else:
            self.output_idxs = [i for i, l in enumerate(self.data.level_names.values) 
                                if any([bool(re.match(o, l)) for o in output_vars])]

        # if bins None then calculate bounds for bins from dataset       
        self.bins_t = np.linspace(self.data.isel(level =self.output_idxs).min(), self.data.isel(level =self.output_idxs).max(), 100) if bins_t is None else bins_t
        # bin data
        self.binned_data = xr.DataArray((np.digitize(self.data.isel(level=self.output_idxs), self.bins_t)-1)[:,:,:,0], dims=['time', 'lat', 'lon'], coords={'time':self.data.time.values, 'lat': self.data.lat.values, 'lon': self.data.lon.values})
        
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std(('time', 'lat', 'lon')).compute() if std is None else std
        # Normalize
        self.data = (self.data - self.mean) / self.std
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        del ds
        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data.load()
        if load: print('Loading data into RAM'); self.binned_data.load()            

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        X = self.data.isel(time=idxs).values
        y = self.binned_data.isel(time=idxs + self.lead_time).values
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)

# 5 day forecast so lead time of 120 hrs
bs=32
lead_time=120
output_vars = ['t_850']

# Create a training and validation data generator. Use the train mean, std and bins for validation as well.
dg_train = DataGenerator(
    ds_train.sel(time=slice('1979', '2013')), var_dict, lead_time, batch_size=bs, load=False, output_vars = output_vars)

dg_valid = DataGenerator(
    ds_valid, var_dict, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, bins_t = dg_train.bins_t, shuffle=False, output_vars = output_vars)

dg_test = DataGenerator(
    ds_test, var_dict, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, bins_t = dg_train.bins_t, shuffle=False, output_vars = output_vars)


bin_values = dg_valid.bins_t

# Read in outputs of individual neural network members trained on different sets of variables. 
# Note these outputs are the expectations of the distributions.
output_avg_wind =  np.expand_dims(np.load('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(block_no) + '_wind_[50, 100, 300, 850, 925, 1000]_preds_cat_t_val.npy'), axis = -1)
output_avg_geo = np.expand_dims(np.load('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(block_no) + '_geo_[500, 600, 700, 850, 925, 1000]_preds_cat_t_val.npy'), axis = -1)
output_avg_temp = np.expand_dims(np.load('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(block_no) + '_temp_[500, 600, 700, 850, 925, 1000]_preds_cat_t_val.npy'),axis = -1)
output_avg_pv = np.expand_dims(np.load('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(block_no) + '_pot_vort_[150, 250, 300, 700, 850]_preds_cat_t_val.npy'), axis = -1)
output_avg_sh = np.expand_dims(np.load('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(block_no) + '_specific_humidity_[150, 200, 600, 700, 850, 925, 1000]_preds_cat_t_val.npy'), axis = -1)
output_avg_const = np.expand_dims(np.load('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(block_no) + '_const_no_l_preds_cat_t_val.npy'), axis = -1)

X1, y1 = dg_valid[0]

for i in range(1, len(dg_valid)):
    X2, y2 = dg_valid[i]
    y1 = np.concatenate((y1, y2)) 

# normalise these outputs
output_avg_geo_mean = (output_avg_geo-float(dg_valid.mean.sel(level = 850)))/float(dg_valid.std.sel(level = 850))
output_avg_temp_mean = (output_avg_temp-float(dg_valid.mean.sel(level = 850)))/float(dg_valid.std.sel(level = 850))
output_avg_pv_mean = (output_avg_pv-float(dg_valid.mean.sel(level = 850)))/float(dg_valid.std.sel(level = 850))
output_avg_sh_mean = (output_avg_sh-float(dg_valid.mean.sel(level = 850)))/float(dg_valid.std.sel(level = 850))
output_avg_const_mean = (output_avg_const-float(dg_valid.mean.sel(level = 850)))/float(dg_valid.std.sel(level = 850))
output_avg_wind_mean = (output_avg_wind-float(dg_valid.mean.sel(level = 850)))/float(dg_valid.std.sel(level = 850))

# For reasons of memory delete unnecessary variables
del output_avg_geo
del output_avg_temp
del output_avg_pv
del output_avg_sh
del output_avg_const
del output_avg_wind

stack_test_list = [output_avg_wind_mean, output_avg_geo_mean, output_avg_temp_mean, output_avg_pv_mean, output_avg_sh_mean, output_avg_const_mean]

from tensorflow.keras.layers import concatenate

def build_stack_model(input_shape, stack_list):
    # concatenate merge output from each model
    input_list = [Input(shape=input_shape) for i in range(len(stack_list))]
    merge = concatenate(input_list)
    x = Dense(36, activation = 'relu')(merge)
    x = Dense(36, activation = 'relu')(x)
    hidden = Dense(100)(x)
    # necessary to reshape the data because softmax only accepts 1d data    
    out = Reshape((32*64, 100), input_shape = (32, 64, 100))(hidden)
    out = Activation('softmax')(out)
    out = Reshape((32, 64, 100), input_shape = (32*64, 100))(out)
    return keras.models.Model(input_list, out)

del dg_valid
del ds_whole
del X1
del X2

del output_avg_geo_mean
del output_avg_temp_mean
del output_avg_pv_mean
del output_avg_sh_mean
del output_avg_const_mean
del output_avg_wind_mean

# build stack model
ensemble_model = build_stack_model((32, 64, 1), stack_test_list)

# use sparse_categorical metrics because our data is not one-hot encoded
ensemble_model.compile(keras.optimizers.Adam(5e-5), loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=10,
                        verbose=1, 
                        mode='auto'
                    )

# reduce learning rate when validation loss plateaus
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss',
            patience=3,
            factor=0.2,
            verbose=1)  

ensemble_model.fit(x = stack_test_list, y = y1, epochs = 300, validation_split = 0.2, shuffle = True, verbose =2,
                  callbacks = [early_stopping_callback, reduce_lr_callback
                    ])


ensemble_model.save_weights('stacked_cat_120_' + str(block_no) + '_train_t.h5')

# read in outputs of individual neural network members on the test dataset
output_test_wind = np.expand_dims(np.load('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(block_no) + '_wind_[50, 100, 300, 850, 925, 1000]_preds_cat_t_test.npy'), axis = -1)
output_test_geo = np.expand_dims(np.load('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(block_no) + '_geo_[500, 600, 700, 850, 925, 1000]_preds_cat_t_test.npy'), axis = -1)
output_test_temp = np.expand_dims(np.load('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(block_no) + '_temp_[500, 600, 700, 850, 925, 1000]_preds_cat_t_test.npy'), axis = -1)
output_test_pv = np.expand_dims(np.load('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(block_no) + '_pot_vort_[150, 250, 300, 700, 850]_preds_cat_t_test.npy'), axis = -1)
output_test_sh = np.expand_dims(np.load('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(block_no) + '_specific_humidity_[150, 200, 600, 700, 850, 925, 1000]_preds_cat_t_test.npy'), axis = -1)
output_test_const = np.expand_dims(np.load('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(block_no) + '_const_no_l_preds_cat_t_test.npy'), axis = -1)

# normalise these outputs
output_test_geo_mean = (output_test_geo-float(dg_test.mean.sel(level = 850)))/float(dg_test.std.sel(level = 850))
output_test_temp_mean = (output_test_temp-float(dg_test.mean.sel(level = 850)))/float(dg_test.std.sel(level = 850))
output_test_pv_mean = (output_test_pv-float(dg_test.mean.sel(level = 850)))/float(dg_test.std.sel(level = 850))
output_test_sh_mean = (output_test_sh-float(dg_test.mean.sel(level = 850)))/float(dg_test.std.sel(level = 850))
output_test_const_mean = (output_test_const-float(dg_test.mean.sel(level = 850)))/float(dg_test.std.sel(level = 850))
output_test_wind_mean = (output_test_wind-float(dg_test.mean.sel(level = 850)))/float(dg_test.std.sel(level = 850))


stack_test_test = [output_test_wind_mean, output_test_geo_mean, output_test_temp_mean, output_test_pv_mean, output_test_sh_mean, output_test_const_mean]

del output_test_geo
del output_test_temp
del output_test_pv
del output_test_sh
del output_test_const
del output_test_wind

# generate stacked network predictions
stack_out = ensemble_model.predict(stack_test_test)

# calculate the expectations of the distributions in order to be able to calculate the error to the real data
fc = np.dot(stack_out, dg_test.bins_t)

fc_conv_ds_avg = xr.Dataset({
        't': xr.DataArray(
              fc,
               dims=['time', 'lat', 'lon'],
               coords={'time':dg_test.data.time[120:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,
                })})
    
cnn_rmse_arg = compute_weighted_rmse(fc_conv_ds_avg, ds_test.t.sel(level = 850)[120:]).compute()
print(cnn_rmse_arg)

np.save('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/5_day_' + str(args.block_no) + '_predictions.npy', stack_out)
