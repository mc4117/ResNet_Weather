import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from src.score import *
import re
from collections import OrderedDict
import itertools

import sys
print("Script name ", sys.argv[0])

block_no = sys.argv[1]

"""
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
"""
 
DATADIR = '/rds/general/user/mc4117/home/WeatherBench/data/'

z500_valid = load_test_data(f'{DATADIR}geopotential_500', 'z')
t850_valid = load_test_data(f'{DATADIR}temperature_850', 't')
valid = xr.merge([z500_valid, t850_valid])

z = xr.open_mfdataset(f'{DATADIR}geopotential_500/*.nc', combine='by_coords')
t = xr.open_mfdataset(f'{DATADIR}temperature_850/*.nc', combine='by_coords').drop('level')

# For the data generator all variables have to be merged into a single dataset.
datasets = [z, t]
ds = xr.merge(datasets)

# In this notebook let's only load a subset of the training data
ds_train = ds.sel(time=slice('1979', '2016'))  
ds_test = ds.sel(time=slice('2017', '2018'))

class DataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True, mean=None, std=None, bins_z = None, bins_t = None):
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
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            try:
                data.append(ds[var].sel(level=levels))
            except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1))

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        # Normalize
        self.data = (self.data - self.mean) / self.std
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time
        
        
        self.bins_z = np.linspace(ds.z.min(), ds.z.max(), 100) if bins_z is None else bins_z
        self.bins_t = np.linspace(ds.t.min(), ds.t.max(), 100) if bins_t is None else bins_t
        
        digitized_z_data = xr.DataArray(np.digitize(ds.z, self.bins_z)-1, dims=['time', 'lat', 'lon'], coords={'time':self.data.time.values, 'lat': self.data.lat.values, 'lon': self.data.lon.values})
        
        digitized_t_data = xr.DataArray(np.digitize(ds.t, self.bins_t)-1, dims=['time', 'lat', 'lon'], coords={'time':self.data.time.values, 'lat': self.data.lat.values, 'lon': self.data.lon.values})        
        self.binned_data = xr.DataArray([digitized_z_data, digitized_t_data], dims=['var', 'time', 'lat', 'lon'], coords={'var': ['z', 't'], 'time':ds.time.values, 'lat': ds.lat.values, 'lon': ds.lon.values}).transpose('time', 'lat', 'lon', 'var')        

        del digitized_z_data
        del digitized_t_data

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
            
# then we need a dictionary for all the variables and levels we want to extract from the dataset
dic = OrderedDict({'z': None, 't': None})

bs=32
lead_time=72

# Create a training and validation data generator. Use the train mean and std for validation as well.
dg_train = DataGenerator(
    ds_train.sel(time=slice('1979', '2015')), dic, lead_time, batch_size=bs, load=True)

dg_valid = DataGenerator(
    ds_train.sel(time=slice('2016', '2016')), dic, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, shuffle=False, bins_z = dg_train.bins_z, bins_t = dg_train.bins_t)

dg_test = DataGenerator(
    ds_test, dic, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, shuffle=False, bins_z = dg_train.bins_z, bins_t = dg_train.bins_t)



class PeriodicPadding2D(tf.keras.layers.Layer):
    def __init__(self, pad_width, **kwargs):
        super().__init__(**kwargs)
        self.pad_width = pad_width

    def call(self, inputs, **kwargs):
        if self.pad_width == 0:
            return inputs
        inputs_padded = tf.concat(
            [inputs[:, :, -self.pad_width:, :], inputs, inputs[:, :, :self.pad_width, :]], axis=2)
        # Zero padding in the lat direction
        inputs_padded = tf.pad(inputs_padded, [[0, 0], [self.pad_width, self.pad_width], [0, 0], [0, 0]])
        return inputs_padded

    def get_config(self):
        config = super().get_config()
        config.update({'pad_width': self.pad_width})
        return config


class PeriodicConv2D(tf.keras.layers.Layer):
    def __init__(self, filters,
                 kernel_size,
                 conv_kwargs={},
                 **kwargs, ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_kwargs = conv_kwargs
        if type(kernel_size) is not int:
            assert kernel_size[0] == kernel_size[1], 'PeriodicConv2D only works for square kernels'
            kernel_size = kernel_size[0]
        pad_width = (kernel_size - 1) // 2
        self.padding = PeriodicPadding2D(pad_width)
        self.conv = Conv2D(
            filters, kernel_size, padding='valid', **conv_kwargs
        )

    def call(self, inputs):
        return self.conv(self.padding(inputs))

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'kernel_size': self.kernel_size, 'conv_kwargs': self.conv_kwargs})
        return config
    
def convblock(inputs, f, k, l2, dr = 0):
    x = inputs
    if l2 is not None:
        x = PeriodicConv2D(f, k, conv_kwargs={
            'kernel_regularizer': keras.regularizers.l2(l2)})(x) 
    else:
        x = PeriodicConv2D(f, k)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    if dr>0: x = Dropout(dr)(x, training = True)

    return x

def build_resnet_cnn(filters, kernels, input_shape, l2 = None, dr = 0, skip = True):
    """Fully convolutional residual network"""

    x = input = Input(shape=input_shape)
    x = convblock(x, filters[0], kernels[0], dr)

    #Residual blocks
    for f, k in zip(filters[1:-1], kernels[1:-1]):
        y = x
        for _ in range(2):
            x = convblock(x, f, k, l2, dr)
        if skip: x = Add()([y, x])

    output = PeriodicConv2D(filters[-1], kernels[-1])(x)
    
    return keras.models.Model(input, output)

filt = [64]
kern = [5]

for i in range(int(block_no)):
    filt.append(64)
    kern.append(5)

filt.append(2)
kern.append(5)

cnn = build_resnet_cnn(filt, kern, (32, 64, 2), l2 = 1e-5)

cnn.compile(keras.optimizers.Adam(5e-5), 'mse')


print(cnn.summary())
"""
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=5,
                        verbose=1, 
                        mode='auto'
                    )

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss',
            patience=2,
            factor=0.2,
            verbose=1)


cnn.fit(dg_train, epochs=100, validation_data=dg_valid, 
          callbacks=[early_stopping_callback, reduce_lr_callback]
         )
"""

cnn.load_weights('/rds/general/user/mc4117/home/WeatherBench/saved_models/whole_cat_both_no_ohe_' + str(block_no) + '.h5')

output = cnn.predict(dg_test)

preds_cat = xr.DataArray(output[:, :, :, 0], dims=['time', 'lat', 'lon'], coords={'time': dg_test.valid_time, 'lat': dg_test.ds.lat, 'lon': dg_test.ds.lon}) 

preds_cat_int = np.round(preds_cat, 0)

nx, ny, nz = preds_cat.shape

preds_cat_inv = np.zeros((nx, ny, nz))

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            preds_cat_inv[i, j, k] = dg_test.bins_z[int(preds_cat_int[i, j, k])]

full_out_ds = xr.DataArray(
        preds_cat_inv,
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon})

print(compute_weighted_rmse(full_out_ds, z500_valid[72:]).compute())


preds_cat_t = xr.DataArray(output[:, :, :, 1], dims=['time', 'lat', 'lon'], coords={'time': dg_test.valid_time, 'lat': dg_test.ds.lat, 'lon': dg_test.ds.lon})

preds_cat_int_t = np.round(preds_cat_t, 0)

nx, ny, nz = preds_cat_t.shape

preds_cat_inv_t = np.zeros((nx, ny, nz))

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            preds_cat_inv_t[i, j, k] = dg_test.bins_t[int(preds_cat_int_t[i, j, k])]

full_out_ds_t = xr.DataArray(
        preds_cat_inv_t,
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon})

print(compute_weighted_rmse(full_out_ds, t850_valid[72:]).compute())



X1, y1 = dg_test[0]

for i in range(1, len(dg_test)):
    X2, y2 = dg_test[i]
    y1 = np.concatenate((y1, y2))

nx_y, ny_y, nz_y = y1.shape

y1_full = np.zeros((nx_y, ny_y, nz_y))

for i in range(nx_y):
    for j in range(ny_y):
        for k in range(nz_y):
            y1_full[i][j][k] = dg_test.bins_z[y1[i][j][k].argmax()]


real_ds = xr.DataArray(
        y1_full[:, :, :, 0],
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon})

print(compute_weighted_rmse(full_out_ds, real_ds).compute())

