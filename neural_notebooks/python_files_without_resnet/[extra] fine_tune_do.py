import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from src.score import *
import re

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

var_dict = {
    'geopotential': ('z', [500]),
    'temperature': ('t', [850]),
#    'specific_humidity': ('q', [850]),
#    '2m_temperature': ('t2m', None),
#    'potential_vorticity': ('pv', [50, 100]),
#    'constants': ['lsm', 'orography']
}

ds = [xr.open_mfdataset(f'{DATADIR}/{var}/*.nc', combine='by_coords') for var in var_dict.keys()]

ds_whole = xr.merge(ds)

ds_train = ds_whole.sel(time=slice('2014', '2015'))
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
    
def build_cnn(filters, kernels, input_shape, dr=0, fine_tune = False):
    """Fully convolutional network"""
    x = input = Input(shape=input_shape)   
    for f, k in zip(filters[:-1], kernels[:-1]):
        if k == 5:
            x = PeriodicConv2D(f, k)(x)  
        else:
            x = PeriodicConv2D(f, k)(x)
            x = PeriodicConv2D(f, k)(x)
        #else:
        #    x = BatchNormalization()(x)   
        x = LeakyReLU()(x)
        if fine_tune:
            x = BatchNormalization()(x, training = True)        
        if dr > 0:
            x = Dropout(dr)(x, training = False) 
    output = PeriodicConv2D(filters[-1], kernels[-1])(x)
    output = PeriodicConv2D(filters[-1], kernels[-1])(x)     
    return keras.models.Model(input, output)

def create_predictions(model, dg):
    """Create non-iterative predictions"""
    preds = xr.DataArray(
        model.predict_generator(dg),
        dims=['time', 'lat', 'lon', 'level'],
        coords={'time': dg.valid_time, 'lat': dg.data.lat, 'lon': dg.data.lon, 
                'level': dg.data.isel(level=dg.output_idxs).level,
                'level_names': dg.data.isel(level=dg.output_idxs).level_names
               },
    )
    # Unnormalize
    preds = (preds * dg.std.isel(level=dg.output_idxs).values + 
             dg.mean.isel(level=dg.output_idxs).values)
    unique_vars = list(set([l.split('_')[0] for l in preds.level_names.values])); unique_vars
    
    das = []
    for v in unique_vars:
        idxs = [i for i, vv in enumerate(preds.level_names.values) if vv.split('_')[0] in v]
        #print(v, idxs)
        da = preds.isel(level=idxs).squeeze().drop('level_names')
        if not 'level' in da.dims: da.drop('level')
        das.append({v: da})
    return xr.merge(das, compat = 'override').drop('level')

# fit with no drop out
cnn = build_cnn([64, 64, 64, 64, 2], [5, 3, 3, 3, 3], (32, 64, 2))
cnn.compile(keras.optimizers.Adam(1e-4), 'mse')

cnn.fit(dg_train, epochs=100, validation_data=dg_valid, 
          callbacks=[tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=2,
                        verbose=1, 
                        mode='auto'
                    )]
         )

# build model with dropout
cnn_2 = build_cnn([64, 64, 64, 64, 2], [5, 3, 3, 3, 3], (32, 64, 2), dr = 0.1, fine_tune = True)

j = 0
i = 0
for k in range(len(cnn_2.layers)):
    if type(cnn_2.layers[j]) == type(cnn.layers[i]):
        print(cnn_2.layers[j])
        print(cnn.layers[i])
        cnn_2.layers[j].set_weights(cnn.layers[i].get_weights())
        j += 1
        i += 1
    elif type(cnn_2.layers[j]) == tf.keras.layers.Dropout:
        j += 1
    elif type(cnn_2.layers[j]) == tf.keras.layers.BatchNormalization:
        j += 1
    else:
        print("error")

cnn_2.compile(keras.optimizers.Adam(1e-5), 'mse')

# fine tune using smaller learning rate with do layers
cnn_2.fit(dg_train, epochs=10, validation_data=dg_valid, 
          callbacks=[tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=2,
                        verbose=1, 
                        mode='auto'
                    )]
         )

# build model without dropout and fine tune

cnn_3 = build_cnn([64, 64, 64, 64, 2], [5, 3, 3, 3, 3], (32, 64, 2), fine_tune = True)

j = 0
i = 0
for k in range(len(cnn_3.layers)):
    if type(cnn_3.layers[j]) == type(cnn.layers[i]):
        cnn_3.layers[j].set_weights(cnn.layers[i].get_weights())
        j += 1
        i += 1
    elif type(cnn_3.layers[j]) == tf.keras.layers.Dropout:
        j += 1
    elif type(cnn_3.layers[j]) == tf.keras.layers.BatchNormalization:
        j += 1
    else:
        print("error")
        
cnn_3.compile(keras.optimizers.Adam(1e-5), 'mse')

# fine tune using smaller learning rate with do layers
cnn_3.fit(dg_train, epochs=10, validation_data=dg_valid, 
          callbacks=[tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=2,
                        verbose=1, 
                        mode='auto'
                    )]
         )

number_of_forecasts = 1

fc = create_predictions(cnn, dg_test)

pred_ensemble=np.ndarray(shape=(2, 17448, 32, 64, number_of_forecasts),dtype=np.float32)

pred2 = np.asarray(fc.to_array(), dtype=np.float32).squeeze()
pred_ensemble[:,:,:,:,0]=pred2
filename_1 = '/rds/general/user/mc4117/ephemeral/saved_pred/no_fine_tune_2.npy'
np.save(filename_1 + '.npy', pred_ensemble)

fc_2 = create_predictions(cnn_2, dg_test)

pred_ensemble=np.ndarray(shape=(2, 17448, 32, 64, number_of_forecasts),dtype=np.float32)

pred2 = np.asarray(fc_2.to_array(), dtype=np.float32).squeeze()
pred_ensemble[:,:,:,:,0]=pred2
filename_2 = '/rds/general/user/mc4117/ephemeral/saved_pred/fine_tune_nodo_2.npy'
np.save(filename_2 + '.npy', pred_ensemble)

fc_3 = create_predictions(cnn_3, dg_test)

pred_ensemble=np.ndarray(shape=(2, 17448, 32, 64, number_of_forecasts),dtype=np.float32)

pred2 = np.asarray(fc_3.to_array(), dtype=np.float32).squeeze()
pred_ensemble[:,:,:,:,0]=pred2

filename_3 = '/rds/general/user/mc4117/ephemeral/saved_pred/fine_tune_do_2.npy'
np.save(filename_3 + '.npy', pred_ensemble)