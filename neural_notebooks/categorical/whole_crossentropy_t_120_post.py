# Consider original data of z500 and t850 with one other variable. Can specify the levels of this one other variable that are included

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
from src.score import *
import re
from collections import OrderedDict

CLI.add_argument(
  "--level_list",
  nargs="*",
  type=int,  # any type/callable can be used here
  default=None,
)

CLI.add_argument(
  "--block_no",
  type = int,
  default = 2,
)

CLI.add_argument(
  "--var_name",
  type = str,
  default = None,
)

args = CLI.parse_args()

var_name = args.var_name
print(args.var_name)
print(args.block_no)

#device_name = tf.test.gpu_device_name()
#if device_name != '/device:GPU:0':
#    raise SystemError('GPU device not found')
#print('Found GPU at: {}'.format(device_name))

 
DATADIR = '/rds/general/user/mc4117/home/WeatherBench/data/'

if args.level_list is not None:
    unique_list = sorted(list(dict.fromkeys(args.level_list)))
else:
    unique_list = None

# For the data generator all variables have to be merged into a single dataset.
if var_name == 'specific_humidity':
    if args.level_list is None:
        unique_list = [150, 200, 600, 700, 850, 925, 1000]
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', [850]),
        'specific_humidity': ('q', unique_list)}
elif var_name == '2m_temp':
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', [850]),
        '2m_temperature': ('t2m', None)}
elif var_name == 'solar_rad':
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', [850]),
        'toa_incident_solar_radiation': ('tisr', None)}
elif var_name == 'pot_vort':
    if args.level_list is None:
        unique_list = [150, 250, 300, 700, 850]
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', [850]),
        'potential_vorticity': ('pv', unique_list)}
elif var_name == 'const':
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', [850]),
        'constants': ['lat2d', 'orography', 'lsm']}
elif var_name == 'wind':
    if args.level_list is None:
        unique_list = [50, 100, 300, 850, 925, 1000]
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', [850]),
        'u_component_of_wind': ('u', unique_list)}    
elif var_name == 'orig':
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', [850])} 
elif var_name == 'temp':
    if args.level_list is None:
        unique_list = [500, 600, 700, 850, 925, 1000]
    else:
        unique_list.append(850)
        unique_list = sorted(list(dict.fromkeys(unique_list)))
    print(unique_list)
    var_dict = {
        'geopotential': ('z', [500]),
        'temperature': ('t', unique_list)}
elif var_name == 'geo':
    if args.level_list is None:
        unique_list = [500, 600, 700, 850, 925, 1000]
    else:
        unique_list.append(500)
        unique_list = sorted(list(dict.fromkeys(unique_list)))
    print(unique_list)
    var_dict = {
        'geopotential': ('z', unique_list),
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

# because missing first values of solar radiation exclude these from the dataset
ds_whole = xr.merge(ds_list).isel(time = slice(7, None))

ds_train = ds_whole.sel(time=slice('1979', '2016'))
ds_valid = ds_train.sel(time=slice('2012', '2016'))
ds_test = ds_whole.sel(time=slice('2017', '2018'))

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

        self.bins_t = np.linspace(self.data.isel(level =self.output_idxs).min(), self.data.isel(level =self.output_idxs).max(), 100) if bins_t is None else bins_t 
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

bs=32
lead_time=120
output_vars = ['t_850']

# Create a training and validation data generator. Use the train mean and std for validation as well.
dg_train = DataGenerator(
    ds_train.sel(time=slice('1979', '2010')), var_dict, lead_time, batch_size=bs, load=True, output_vars = output_vars)

dg_valid = DataGenerator(
    ds_train.sel(time=slice('2012', '2016')), var_dict, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, bins_t = dg_train.bins_t, shuffle=False, output_vars = output_vars)

#dg_valid2 = DataGenerator(
#    ds_train.sel(time=slice('2011', '2011')), var_dict, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, bins_t = dg_train.bins_t, shuffle=False, output_vars = output_vars)

dg_test = DataGenerator(
    ds_test, var_dict, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, bins_t = dg_train.bins_t, shuffle=False, output_vars = output_vars)

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
            filters, kernel_size, padding='valid', dtype = 'float32', **conv_kwargs
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
    x = LeakyReLU(dtype = "float32")(x)
    x = BatchNormalization(dtype = "float32")(x)
    if dr>0: x = Dropout(dr, dtype = "float32")(x, training = True)

    return x

def build_resnet_cnn(filters, kernels, input_shape, l2 = None, dr = 0, skip = True):
    """Fully convolutional residual network"""

    x = input = keras.layers.Input(shape=input_shape)
    x = convblock(x, filters[0], kernels[0], dr)

    #Residual blocks
    for f, k in zip(filters[1:-1], kernels[1:-1]):
        y = x
        for _ in range(2):
            x = convblock(x, f, k, l2, dr)
        if skip: x = Add()([y, x])
    out = Reshape((32*64, 100), input_shape = (32, 64, 100))(x)
    out = Activation('softmax')(out)
    out = Reshape((32, 64, 100), input_shape = (32*64, 100))(out)
   
    return keras.models.Model(input, out)

filt = [100]
kern = [5]

for i in range(int(args.block_no)):
    filt.append(100)
    kern.append(5)

filt.append(1)
kern.append(5)

if unique_list is not None:
    if var_name == "temp":
        tot_var = 1 + len(unique_list)
    elif var_name == "geo":
        tot_var = 1 + len(unique_list)
    else:
        tot_var = 2 + len(unique_list)
else:
    if var_name == "const":
        tot_var = 5
    else:
        tot_var = len(var_dict)
    unique_list = str('no_l')

cnn = build_resnet_cnn(filt, kern, (32, 64, tot_var), l2 = 1e-5, dr = 0.1)

cnn.compile(keras.optimizers.Adam(5e-5), loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])

print(cnn.summary())

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=3,
                        verbose=1, 
                        mode='auto'
                    )

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss',
            patience=2,
            factor=0.2,
            verbose=1)

"""
cnn.fit(dg_train, epochs=100, validation_data=dg_valid2, callbacks=[early_stopping_callback, reduce_lr_callback])
"""

filename = '/rds/general/user/mc4117/ephemeral/saved_models/whole_cat_t_indiv_val_120_' + str(args.block_no) + '_' + str(var_name)
cnn.load_weights(filename + '.h5')    

del dg_train
del ds_whole

no_of_forecasts = 32

#fc_all = []

output_total = 0

for i in range(no_of_forecasts):
    print(i)
    
    fc = np.dot(cnn.predict(dg_valid), dg_valid.bins_t)
    print(fc.shape)
    #fc_all.append(fc)
    output_total += fc
    print(compute_weighted_rmse(output_total/(i+1), ds_valid.t.sel(level = 850)[72:]).compute())
    
output_avg = output_total/no_of_forecasts

np.save('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(args.block_no) + '_' + str(var_name) + '_' + str(unique_list) + '_preds_cat_t_val.npy', output_avg)

del output_avg
del output_total
del fc

output_total = 0

for i in range(no_of_forecasts):
    print(i)
    
    fc = np.dot(cnn.predict(dg_test), dg_test.bins_t)

    output_total += fc.copy()
    
output_avg = output_total/no_of_forecasts
    
np.save('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/120_' + str(args.block_no) + '_' + str(var_name) + '_' + str(unique_list) + '_preds_cat_t_test.npy', output_avg)
