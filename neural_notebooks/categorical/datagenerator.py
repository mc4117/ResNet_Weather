import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
import pickle
from src.score import *
from collections import OrderedDict

import sys
print("Script name ", sys.argv[0])

block_no = sys.argv[1]


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


DATADIR = '/rds/general/user/mc4117/home/WeatherBench/data/'

z500_valid = load_test_data(f'{DATADIR}geopotential_500', 'z')
t850_valid = load_test_data(f'{DATADIR}temperature_850', 't')
valid = xr.merge([z500_valid, t850_valid])

z = xr.open_mfdataset(f'{DATADIR}geopotential_500/*.nc', combine='by_coords')
t = xr.open_mfdataset(f'{DATADIR}temperature_850/*.nc', combine='by_coords').drop('level')

datasets = [z, t]
ds = xr.merge(datasets)


ds_train = ds.sel(time=slice('2014', '2016'))  
ds_test = ds.sel(time=slice('2017', '2018'))

class DataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True, mean=None, std=None):
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
        generic_cat = xr.DataArray([1], coords={'cat': [1]}, dims=['cat'])
        for var, levels in var_dict.items():
            try:
                data.append(ds[var].sel(level=levels).expand_dims({'cat': generic_cat}, 1))
            except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1).expand_dims({'cat': generic_cat}, 1))

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level', 'cat')
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        # Normalize
        self.data = (self.data - self.mean) / self.std
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time
        
        
        bins_z = np.linspace(ds.z.min(), ds.z.max(), 100)
        digitized_z = np.digitize(ds.z, bins_z)-1
        
        bins_t = np.linspace(ds.t.min(), ds.t.max(), 100)
        digitized_t = np.digitize(ds.t, bins_t)-1
        


        binned_data_z = to_categorical(digitized_z, num_classes = 100, dtype = 'int8')
        binned_data_t = to_categorical(digitized_t, num_classes = 100, dtype = 'int8')        

        self.binned_data_z = xr.DataArray(
            binned_data_z, dims=['time', 'lat', 'lon', 'cat'],
            coords={'time':ds.time.values, 'lat': ds.lat.values, 'lon': ds.lon.values, 'cat': range(100)})  
        
        self.binned_data_t = xr.DataArray(
            binned_data_t, dims=['time', 'lat', 'lon', 'cat'],
            coords={'time':ds.time.values, 'lat': ds.lat.values, 'lon': ds.lon.values, 'cat': range(100)})

        self.binned_data = xr.DataArray([self.binned_data_z, self.binned_data_t], 
            dims=['var', 'time', 'lat', 'lon', 'cat'],
            coords={'var': ['z', 't'], 'time':ds.time.values, 'lat': ds.lat.values, 'lon': ds.lon.values, 'cat': range(100)}).transpose('time', 'lat', 'lon', 'var', 'cat')


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

dic = OrderedDict({'z': None, 't': None})

bs=32
lead_time=72


# Create a training and validation data generator. Use the train mean and std for validation as well.
dg_train = DataGenerator(
    ds_train.sel(time=slice('1979', '2015')), dic, lead_time, batch_size=bs, load=True)

dg_valid = DataGenerator(
    ds_train.sel(time=slice('2016', '2016')), dic, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, shuffle=False)

dg_test = DataGenerator(
    ds_test, dic, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, shuffle=False)

class PeriodicPadding2D(tf.keras.layers.Layer):
    def __init__(self, pad_width, **kwargs):
        super().__init__(**kwargs)
        self.pad_width = pad_width

    def call(self, inputs, **kwargs):
        if self.pad_width == 0:
            return inputs
        inputs_padded = tf.concat(
            [inputs[:, :, -self.pad_width:, :, :], inputs, inputs[:, :, :self.pad_width, :, :]], axis=2)
        # Zero padding in the lat direction
        inputs_padded = tf.pad(inputs_padded, [[0, 0], [self.pad_width, self.pad_width], [0, 0], [0, 0], [0,0]])
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
        pad_width = (kernel_size[1] - 1) // 2
        self.padding = PeriodicPadding2D(pad_width)
        self.conv = Conv3D(
            filters, kernel_size, padding='valid', **conv_kwargs)
        

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

    output_conv = PeriodicConv2D(filters[-1], kernels[-1])(x)
    
    output = Softmax()(output_conv)
    
    return keras.models.Model(input, output)

filt = [64]
kern = [[5, 5, 1]]

for i in range(int(block_no)):
    filt.append(64)
    kern.append([5, 5, 1])

filt.append(100)
kern.append([5, 5, 1])


cnn = build_resnet_cnn(filt, kern, (32, 64, 2, 1))
cnn.compile(keras.optimizers.Adam(1e-4), 'mse')

print(cnn.summary())


cnn.compile(keras.optimizers.Adam(5e-5), 'mse')

print(cnn.summary())

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

cnn.save_weights('/rds/general/user/mc4117/home/WeatherBench/saved_models/whole_cat_res_both_' + str(block_no) + '.h5')

