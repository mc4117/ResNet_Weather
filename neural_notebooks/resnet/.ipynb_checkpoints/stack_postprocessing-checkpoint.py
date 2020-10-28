import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from src.score import *
import re

import generate_data as gd

# resnet/whole_mm_indiv_data.py using 5 blocks

DATADIR = '/rds/general/user/mc4117/home/WeatherBench/data/'

z500_valid = load_test_data(f'{DATADIR}geopotential_500', 'z')
t850_valid = load_test_data(f'{DATADIR}temperature_850', 't')
valid = xr.merge([z500_valid, t850_valid])

# For the data generator all variables have to be merged into a single dataset.
var_dict = {
    'geopotential': ('z', [500]),
    'temperature': ('t', [850]),
}

# For the data generator all variables have to be merged into a single dataset.
ds = [xr.open_mfdataset(f'{DATADIR}/{var}/*.nc', combine='by_coords') for var in var_dict.keys()]
ds_whole = xr.merge(ds, compat = 'override')

# load all training data
ds_test = ds_whole.sel(time=slice('2017', '2018'))

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
            
import re

bs=32
lead_time=72
output_vars = ['z_500', 't_850']

txt = open('dg_train.txt').read()
output = [i.split(': ') for i in txt.split('\n')]

output_mean = [float(output[0][1]), float(output[1][1])]
output_std = [float(output[2][1]), float(output[3][1])]

# Now also a generator for testing. Impartant: Shuffle must be False!
dg_test = DataGenerator(ds_test, var_dict, lead_time, batch_size=bs, mean=output_mean, std=output_std,
                         shuffle=False, output_vars=output_vars)

pred_ensemble_1 = np.load('/rds/general/user/mc4117/ephemeral/saved_pred/whole_res_indiv_data_do_5_specific_humidity.npy')
pred_ensemble_2 = np.load('/rds/general/user/mc4117/ephemeral/saved_pred/whole_res_indiv_data_do_5_2m temp.npy')
#pred_ensemble_3 = np.load('/rds/general/user/mc4117/ephemeral/saved_pred/whole_res_indiv_data_do_5_solar rad.npy')
pred_ensemble_4 = np.load('/rds/general/user/mc4117/ephemeral/saved_pred/whole_res_indiv_data_do_5_pot_vort.npy')
pred_ensemble_5 = np.load('/rds/general/user/mc4117/ephemeral/saved_pred/whole_res_indiv_data_do_5_const.npy')
#pred_ensemble_6 = np.load('/rds/general/user/mc4117/ephemeral/saved_pred/whole_res_indiv_data_do_5_orig.npy')

samples = 20
preds_sh = xr.Dataset({
    'z': xr.DataArray(
        pred_ensemble_1[0, ...],
        dims=['time', 'lat', 'lon', 'ens'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon, 'ens': np.arange(samples), 
                },
    ),
    't': xr.DataArray(
        pred_ensemble_1[1, ...],
        dims=['time', 'lat', 'lon', 'ens'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon, 'ens': np.arange(samples), 
                },
    )
})


preds_2mtemp = xr.Dataset({
    'z': xr.DataArray(
        pred_ensemble_2[0, ...],
        dims=['time', 'lat', 'lon', 'ens'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon, 'ens': 20+ np.arange(samples), 
                },
    ),
    't': xr.DataArray(
        pred_ensemble_2[1, ...],
        dims=['time', 'lat', 'lon', 'ens'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon, 'ens': 20+ np.arange(samples), 
                },
    )
})


preds_pv = xr.Dataset({
    'z': xr.DataArray(
        pred_ensemble_4[0, ...],
        dims=['time', 'lat', 'lon', 'ens'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon, 'ens': 40+ np.arange(samples), 
                },
    ),
    't': xr.DataArray(
        pred_ensemble_4[1, ...],
        dims=['time', 'lat', 'lon', 'ens'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon, 'ens': 40 + np.arange(samples), 
                },
    )
})

preds_const = xr.Dataset({
    't': xr.DataArray(
        pred_ensemble_5[0, ...],
        dims=['time', 'lat', 'lon', 'ens'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon, 'ens': 60+ np.arange(samples), 
                },
    ),
    'z': xr.DataArray(
        pred_ensemble_5[1, ...],
        dims=['time', 'lat', 'lon', 'ens'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon, 'ens': 60 + np.arange(samples), 
                },
    )
})

"""
preds_orig = xr.Dataset({
    'z': xr.DataArray(
        pred_ensemble_6[0, ...],
        dims=['time', 'lat', 'lon', 'ens'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon, 'ens': 80+ np.arange(20), 
                },
    ),
    't': xr.DataArray(
        pred_ensemble_6[1, ...],
        dims=['time', 'lat', 'lon', 'ens'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon, 'ens': 80 + np.arange(20), 
                },
    )
})


preds_sr = xr.Dataset({
    'z': xr.DataArray(
        pred_ensemble_3[0, ...],
        dims=['time', 'lat', 'lon', 'ens'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon, 'ens': 100 + np.arange(samples), 
                },
    ),
    't': xr.DataArray(
        pred_ensemble_3[1, ...],
        dims=['time', 'lat', 'lon', 'ens'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon, 'ens': 100 + np.arange(samples), 
                },
    )
})
"""

X1, y1 = dg_test[0]

for i in range(1, len(dg_test)):
    X2, y2 = dg_test[i]
    y1 = np.concatenate((y1, y2))  

mean = output_mean
std = output_std

stack_test_list = []

for i in range(12):
    stack_test_list.append((np.transpose(preds_sh.isel(ens = i).to_array().data, axes = [1, 2, 3, 0]) - mean)/std)
for i in range(12):
    stack_test_list.append((np.transpose(preds_2mtemp.isel(ens = i).to_array().data, axes = [1, 2, 3, 0]) - mean)/std)
for i in range(12):
    stack_test_list.append((np.transpose(preds_pv.isel(ens = i).to_array().data, axes = [1, 2, 3, 0]) - mean)/std)
for i in range(12):
    stack_test_list.append((np.transpose(preds_const.isel(ens = i).to_array().data, axes = [1, 2, 3, 0]) - mean)/std)
#for i in range(len(preds_sr.ens)):
#    stack_test_list.append((np.transpose(preds_sr.isel(ens = i).to_array().data, axes = [1, 2, 3, 0]) - mean)/std)
#for i in range(len(preds_orig.ens)):
#    stack_test_list.append((np.transpose(preds_orig.isel(ens = i).to_array().data, axes = [1, 2, 3, 0]) - mean)/std)    
from tensorflow.keras.layers import concatenate

#def my_init(shape, dtype=None):
#    print(shape)
#    return tf.ones(shape, dtype=dtype)/48

def build_stack_model(input_shape, stack_list):
    # concatenate merge output from each model
    input_list = [Input(shape=input_shape) for i in range(len(stack_list))]
    merge = concatenate(input_list)
    hidden = Dense(48, activation='relu')(merge)
    output = Dense(2)(hidden)
    return keras.models.Model(input_list, output)

ensemble_model = build_stack_model((32, 64, 2), stack_test_list)

ensemble_model.compile(keras.optimizers.Adam(1e-4), 'mse')

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

ensemble_model.fit(x = stack_test_list, y = y1, epochs = 800, validation_split = 0.2, shuffle = True
                  , callbacks = [early_stopping_callback, reduce_lr_callback
                    ])

ensemble_model.save_weights('/rds/general/user/mc4117/home/WeatherBench/saved_models/stacked_model_new.h5')
