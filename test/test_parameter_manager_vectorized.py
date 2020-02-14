import numpy as np
import pandas as pd
import xarray as xr
import gribmanager as gm
import gribmanager.grib_keys as gk
import gribmanager.parameter_manager as pm_d


INPUTs = ['/home/wolp/data/ECMWF/EN19090900']#, '/home/wolp/data/ECMWF/EN19090903']
p_m = gm.ParameterManager(INPUTs[0])
p_m_d = pm_d.ParameterManager(INPUTs[0])
temp = p_m.get_parameter(130)
sp = p_m.get_parameter(gk.SURFACE_PRESSURE_PARAM_ID)
temp_d = p_m_d.get_parameter(130)
sp_d = p_m_d.get_parameter(gk.SURFACE_PRESSURE_PARAM_ID)
#spv = pmv.HorizontalParameter(sp._parameter)
#tempv = pmv.VerticalParameterInModelLevel(temp._parameter_at_all_levels, spv)

n = 1000
lat = xr.DataArray(np.random.uniform(low=-90, high=90, size=n), dims='t')
lon = xr.DataArray(np.random.uniform(low=-180, high=180, size=n), dims='t')
pressure = xr.DataArray(np.random.uniform(low=1000, high=40000, size=n), dims='t')

v = temp.interp(lat=lat, lon=lon, pressure=pressure)
v_np = temp.interp_numpy(lat=lat, lon=lon, pressure=pressure)

coords = xr.Dataset({'lat': lat, 'lon': lon, 'pressure': pressure})

coords_df = coords.to_dataframe()
v1 = coords_df.apply(lambda row: temp_d.get_value_at(row.lat, row.lon, row.pressure), axis='columns')

v2 = pd.Series(index=range(1000))
for t, lat, lon, pressure in coords_df.itertuples(name='coords'):
    v2.loc[t] = temp_d.get_value_at(lat, lon, pressure)
