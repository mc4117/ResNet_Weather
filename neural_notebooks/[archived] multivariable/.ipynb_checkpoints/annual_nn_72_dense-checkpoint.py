import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from src.score import *
import re

import sys
print("Script name ", sys.argv[0])

var_name = sys.argv[1]

print(var_name)

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

class DataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True, 
                 mean=None, std=None, output_vars=None):
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
            self.output_idxs = range(len(dg_valid.data.level))
        else:
            self.output_idxs = [i for i, l in enumerate(self.data.level_names.values) 
                                if any([bool(re.match(o, l)) for o in output_vars])]
        
        # Normalize
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
#         self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        self.std = self.data.std(('time', 'lat', 'lon')).compute() if std is None else std
        self.data = (self.data - self.mean) / self.std
        
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        X = self.data.isel(time=idxs).values
        y = self.data.isel(time=idxs + self.lead_time, level=self.output_idxs).values
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)

DATADIR = '/rds/general/user/mc4117/home/WeatherBench/data/'

if var_name == 'specific_humidity':
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', [850]),
        'specific_humidity': ('q', [500, 850])}
elif var_name == '2m temp':
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', [850]),
        '2m_temperature': ('t2m', None)}
elif var_name == 'solar rad':
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', [850]),
        'toa_incident_solar_radiation': ('tisr', None)}
elif var_name == 'pot_vort':
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', [850]),
        'potential_vorticity': ('pv', [500, 850])}
elif var_name == 'const':
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', [850]),
        'constants': ['lat2d', 'orography', 'lsm']}
elif var_name == 'orig':
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', [850])} 

ds = [xr.open_mfdataset(f'{DATADIR}/{var}/*.nc', combine='by_coords') for var in var_dict.keys()]

ds_whole = xr.merge(ds)

ds_train = ds_whole.sel(time=slice('2015', '2015'))
ds_valid = ds_whole.sel(time=slice('2016', '2016'))
ds_test = ds_whole.sel(time=slice('2017', '2018'))

bs=32
lead_time=72
output_vars = ['z_500', 't_850']

# Create a training and validation data generator. Use the train mean and std for validation as well.
dg_train = DataGenerator(ds_train, var_dict, lead_time, batch_size=bs, load=True, 
                         output_vars=output_vars)
dg_valid = DataGenerator(ds_valid, var_dict, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, 
                         shuffle=False, output_vars=output_vars)

dg_test = DataGenerator(ds_test, var_dict, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, 
                         shuffle=False, output_vars=output_vars)

X, y = dg_train[0]; 

print(X.shape)
print(y.shape)

def build_dnn(input_shape):
    """Fully convolutional network"""
    x = input = Input(shape=input_shape)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    output = Dense(2)(x)
    return keras.models.Model(input, output)

if var_name == 'specific_humidity':
    cnn = build_dnn((32, 64, 4))
elif var_name == 'pot_vort':
    cnn = build_dnn((32, 64, 4))
elif var_name == 'const':
    cnn = build_dnn((32, 64, 5))
elif var_name == 'orig':
    cnn = build_dnn((32, 64, 2))
else:
    cnn = build_dnn((32, 64, 3))

cnn.compile(keras.optimizers.Adam(0.001), 'mse')

print(cnn.summary())

def scheduler(epoch, lr):
    if epoch < 3:
        return lr/5
    elif epoch < 6:
        return lr/25
    elif epoch < 9:
        return lr/(25*5)
    elif epoch < 12:
        return lr/625
    elif epoch < 15:
        return lr/(625*5)
    else:
        return lr/(625*25)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)    

cnn.fit(x = dg_train, epochs=18, validation_data=dg_valid, callbacks=[callback])

cnn.save_weights('/rds/general/user/mc4117/home/WeatherBench/saved_models/72_gpu_annual_dense_' + str(var_name) + '.h5')