import gribmanager as gm
import glob
import xarray as xr


#on NUWA:
#enfiles = glob.glob('/o3p/ECMWF/ENFILES/EN1906*')
#iagos_file = '/o3p/iagos/iagosv2/netcdf/L2/201906/IAGOS_timeseries_2019061914230591.nc'

enfiles = glob.glob('/home/wolp/data/ECMWF/EN1906*')
iagos_file = '/home/wolp/data/IAGOS/IAGOS_timeseries_2019061914230591.nc'

iagos_df = xr.open_dataset(iagos_file).to_dataframe()
interpolation_manager = gm.InterpolationInTimeManager(enfiles, iagos_df)
param_specs = [gm.ParameterSpecification(id=130, name='temp', param_id=130,
                                         filter=lambda x: x[gm.TYPE_OF_LEVEL] == gm.HYBRID_LEVEL_TYPE),
               gm.ParameterSpecification(id=77, name='etadot', param_id=77,
                                         filter=lambda x: x[gm.TYPE_OF_LEVEL] == gm.HYBRID_LEVEL_TYPE),
               gm.ParameterSpecification(id=129, name='z', param_id=129,
                                         filter=lambda x: x[gm.LEVEL] == 500 and
                                                          x[gm.TYPE_OF_LEVEL] == gm.ISOBARIC_IN_HPA_LEVEL_TYPE,
                                         must_be_unique=True),
               gm.ParameterSpecification(id=134, name='surf-p', param_id=134, must_be_unique=True),
               gm.ParameterSpecification(id=159, name='zPBL', param_id=159, must_be_unique=True),
               gm.ParameterSpecification(id=22802, name='orography', param_id=129,
                                         filter=lambda x: x[gm.LEVEL] == 0 and
                                                          x[gm.TYPE_OF_LEVEL] == gm.SURFACE_LEVEL,
                                         must_be_unique=True)]

pva_df = interpolation_manager.process_iagos_file(param_specs)
iagos_df_join_pva_df = iagos_df.join(pva_df)
