import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import xarray as xr
import re

DATADIR = '/rds/general/user/mc4117/home/WeatherBench/data/'

var_dict = {
    'geopotential': ('z', [500, 850]),
    'temperature': ('t', [500, 850]),
    'specific_humidity': ('q', [850]),
    '2m_temperature': ('t2m', None),
    'potential_vorticity': ('pv', [50, 100]),
    'constants': ['lsm', 'orography']
}

print(var_dict)

"""
class DataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True, 
                 mean=None, std=None, output_vars=None):

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
"""


ds_list = []

for long_var, params in var_dict.items():
    if long_var == 'constants':
        print(long_var)
        data = xr.open_mfdataset(f'{DATADIR}/{long_var}/*.nc', combine='by_coords')
        data['lat'] = data['lat'].astype('float32')
        data['lon'] = data['lon'].astype('float32')
        ds_list.append(data)
    else:
        print(long_var)
        var, levels = params
        if levels is not None:
            data = xr.open_mfdataset(f'{DATADIR}/{long_var}/*.nc', combine='by_coords').sel(level = levels)
            data['lat'] = data['lat'].astype('float32')
            data['lon'] = data['lon'].astype('float32')
            ds_list.append(data)
        else:
            data = xr.open_mfdataset(f'{DATADIR}/{long_var}/*.nc', combine='by_coords')
            data['lat'] = data['lat'].astype('float32')
            data['lon'] = data['lon'].astype('float32')
            ds_list.append(data)

ds_whole = xr.merge(ds_list)

ds_train = ds_whole.sel(time=slice('1979', '2015'))
ds_valid = ds_whole.sel(time=slice('2016', '2016'))
ds_test = ds_whole.sel(time=slice('2017', '2018'))

bs=32
lead_time=72
output_vars = ['z_500', 't_850']

print('data generator')

# Create a training and validation data generator. Use the train mean and std for validation as well.
#dg_train = DataGenerator(ds_train, var_dict, lead_time, batch_size=bs, load=True, 
#                         output_vars=output_vars)
#dg_valid = DataGenerator(ds_valid, var_dict, lead_time, batch_size=bs, mean=dg_train.mean, std=dg_train.std, 
#                         shuffle=False, output_vars=output_vars)
