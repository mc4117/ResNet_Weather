import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from src.score import *
import re


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
ds_train = ds_whole.sel(time=slice('1979', '2016'))
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
        #if load: print('Loading data into RAM'); self.data.load()

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
            
bs=32
lead_time=72
output_vars = ['z_500', 't_850']

# Create a training and validation data generator. Use the train mean and std for validation as well.
dg_train = DataGenerator(
    ds_train.sel(time=slice('1979', '2013')), var_dict, lead_time, batch_size=bs, load=True, output_vars = output_vars)

#dg_valid2 = DataGenerator(
#    ds_train.sel(time=slice('2015', '2016')), var_dict, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, shuffle=False, output_vars = output_vars)

dg_valid = DataGenerator(
    ds_train.sel(time=slice('2015', '2016')), var_dict, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, shuffle=False, output_vars = output_vars)

# Now also a generator for testing. Impartant: Shuffle must be False!
dg_test = DataGenerator(ds_test, var_dict, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std,
                         shuffle=False, output_vars=output_vars)


X1, y1 = dg_valid[0]

for i in range(1, len(dg_valid)):
    X2, y2 = dg_valid[i]
    X1 = np.concatenate((X1, X2))
    y1 = np.concatenate((y1, y2)) 

real_unnorm =y1* dg_valid.std.isel(level=dg_valid.output_idxs).values+dg_test.mean.isel(level=dg_valid.output_idxs).values

real_ds = xr.Dataset({
    'z': xr.DataArray(
        real_unnorm[..., 0],
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_valid.data.time[72:], 'lat': dg_valid.data.lat, 'lon': dg_valid.data.lon,
                },
    ),
    't': xr.DataArray(
        real_unnorm[..., 1],
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_valid.data.time[72:], 'lat': dg_valid.data.lat, 'lon': dg_valid.data.lon,
                },
    )
})

# read in outputs

temp_levels = xr.open_dataset('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/9_temp_[300, 400, 500, 600, 700, 850]_preds_newval.nc')
geo_levels = xr.open_dataset('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/9_geo_[300, 400, 500, 600, 700, 850]_preds_newval.nc')

sh = xr.open_dataset('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/9_specific_humidity_[300, 500, 600, 700, 850, 925, 1000]_preds_newval.nc')
pv = xr.open_dataset('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/9_pot_vort_[150, 250, 300, 700, 850]_preds_newval.nc')
const = xr.open_dataset('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/9_const_None_preds_newval.nc')

pv_rearranged = xr.Dataset({
    'z': xr.DataArray(
        pv.z.values,
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_valid.data.time[72:], 'lat': dg_valid.data.lat, 'lon': dg_valid.data.lon,
                },
    ),
    't': xr.DataArray(
        pv.t.values,
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_valid.data.time[72:], 'lat': dg_valid.data.lat, 'lon': dg_valid.data.lon,
                },
    )
})

temp_levels_rearranged = xr.Dataset({
    'z': xr.DataArray(
        temp_levels.z.values,
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_valid.data.time[72:], 'lat': dg_valid.data.lat, 'lon': dg_valid.data.lon,
                },
    ),
    't': xr.DataArray(
        temp_levels.t.values,
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_valid.data.time[72:], 'lat': dg_valid.data.lat, 'lon': dg_valid.data.lon,
                },
    )
})

mean_z_t = dg_test.mean.isel(level=dg_test.output_idxs).values
std_z_t = dg_test.std.isel(level=dg_test.output_idxs).values

temp_norm = (np.transpose(temp_levels_rearranged.to_array().data, axes = [1, 2, 3, 0]) - mean_z_t)/std_z_t
geo_norm = (np.transpose(geo_levels.to_array().data, axes = [1, 2, 3, 0]) - mean_z_t)/std_z_t

sh_norm = (np.transpose(sh.to_array().data, axes = [1, 2, 3, 0])-mean_z_t)/std_z_t
pv_norm = (np.transpose(pv_rearranged.to_array().data, axes = [1, 2, 3, 0])-mean_z_t)/std_z_t
const_norm = (np.transpose(const.to_array().data, axes = [1, 2, 3, 0])-mean_z_t)/std_z_t

stack_test_list = [temp_norm, geo_norm, sh_norm, pv_norm, const_norm]

from tensorflow.keras.layers import concatenate

def my_init(shape, dtype=None):
    print(shape)
    return tf.ones(shape, dtype=dtype)/6

def build_stack_model(input_shape, stack_list):
    # concatenate merge output from each model
    input_list = [Input(shape=input_shape) for i in range(len(stack_list))]
    merge = concatenate(input_list)
    hidden = Dense(25, activation='relu', kernel_initializer = my_init)(merge)
    normalize2 = BatchNormalization()(hidden)
    output = Dense(2)(normalize2)
    return keras.models.Model(input_list, output)

ensemble_model = build_stack_model((32, 64, 2), stack_test_list)

ensemble_model.compile(keras.optimizers.Adam(1e-4), 'mse')

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=10,
                        verbose=1, 
                        mode='auto'
                    )

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss',
            patience=3,
            factor=0.2,
            verbose=1)  

ensemble_model.fit(x = stack_test_list, y = y1, epochs = 600, validation_split = 0.2, shuffle = True
                  , callbacks = [early_stopping_callback, reduce_lr_callback
                    ])

ensemble_model.save_weights('stacked_val_comb_9_2.h5')

X1, y1_test = dg_test[0]

for i in range(1, len(dg_test)):
    X2, y2 = dg_test[i]
    X1 = np.concatenate((X1, X2))
    y1_test = np.concatenate((y1_test, y2)) 

real_unnorm_test =y1_test* dg_test.std.isel(level=dg_test.output_idxs).values+dg_test.mean.isel(level=dg_test.output_idxs).values

real_ds_test = xr.Dataset({
    'z': xr.DataArray(
        real_unnorm_test[..., 0],
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,
                },
    ),
    't': xr.DataArray(
        real_unnorm_test[..., 1],
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,
                },
    )
})

# read in outputs

temp_levels_test = xr.open_dataset('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/9_temp_[300, 400, 500, 600, 700, 850]_preds_newtest.nc')
geo_levels_test = xr.open_dataset('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/9_geo_[300, 400, 500, 600, 700, 850]_preds_newtest.nc')

sh_test = xr.open_dataset('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/9_specific_humidity_[300, 500, 600, 700, 850, 925, 1000]_preds_newtest.nc')
pv_test = xr.open_dataset('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/9_pot_vort_[150, 250, 300, 700, 850]_preds_newtest.nc')
const_test = xr.open_dataset('/rds/general/user/mc4117/home/WeatherBench/saved_pred_data/9_const_None_preds_newtest.nc')

pv_rearranged_test = xr.Dataset({
    'z': xr.DataArray(
        pv_test.z.values,
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,
                },
    ),
    't': xr.DataArray(
        pv_test.t.values,
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,
                },
    )
})

temp_levels_rearranged_test = xr.Dataset({
    'z': xr.DataArray(
        temp_levels_test.z.values,
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,
                },
    ),
    't': xr.DataArray(
        temp_levels_test.t.values,
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,
                },
    )
})

geo_levels_rearranged_test = xr.Dataset({
    'z': xr.DataArray(
        geo_levels_test.z.values,
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,
                },
    ),
    't': xr.DataArray(
        geo_levels_test.t.values,
        dims=['time', 'lat', 'lon'],
        coords={'time':dg_test.data.time[72:], 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,
                },
    )
})

temp_test_norm = (np.transpose(temp_levels_rearranged_test.to_array().data, axes = [1, 2, 3, 0]) - mean_z_t)/std_z_t
geo_test_norm = (np.transpose(geo_levels_rearranged_test.to_array().data, axes = [1, 2, 3, 0])- mean_z_t)/std_z_t

sh_test_norm = (np.transpose(sh_test.to_array().data, axes = [1, 2, 3, 0])-mean_z_t)/std_z_t
pv_test_norm = (np.transpose(pv_rearranged_test.to_array().data, axes = [1, 2, 3, 0])-mean_z_t)/std_z_t
const_test_norm = (np.transpose(const_test.to_array().data, axes = [1, 2, 3, 0])-mean_z_t)/std_z_t

stack_test_data_list = [temp_test_norm, geo_test_norm, sh_test_norm, pv_test_norm, const_test_norm]

fc_test = ensemble_model.predict(stack_test_data_list)
preds_un = xr.DataArray(
        fc_test,
        dims=['time', 'lat', 'lon', 'level'],
        coords={'time': dg_test.valid_time, 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,
                'level': dg_test.data.isel(level=dg_test.output_idxs).level,
                'level_names': dg_test.data.isel(level=dg_test.output_idxs).level_names
               },
)

# Unnormalize
preds = preds_un * std_z_t + mean_z_t
unique_vars = list(set([l.split('_')[0] for l in preds.level_names.values])); unique_vars

das = []
for v in unique_vars:
        idxs = [i for i, vv in enumerate(preds.level_names.values) if vv.split('_')[0] in v]
        #print(v, idxs)
        da = preds.isel(level=idxs).squeeze().drop('level_names')
        if not 'level' in da.dims: da.drop('level')
        das.append({v: da})
fc_unnorm_test = xr.merge(das, compat = 'override').drop('level')

print(compute_weighted_rmse(fc_unnorm_test, real_ds_test).compute())

print(compute_weighted_rmse((temp_levels_test + geo_levels_test + sh_test + const_test + pv_test)/5, real_ds_test))

# individual 
print(compute_weighted_rmse(temp_levels_test, real_ds_test))
print(compute_weighted_rmse(geo_levels_test, real_ds_test))
print(compute_weighted_rmse(sh_test, real_ds_test))
print(compute_weighted_rmse(const_test, real_ds_test))
print(compute_weighted_rmse(pv_test, real_ds_test))
