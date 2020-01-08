import pandas as pd
from collections import deque, namedtuple

from gribmanager_obsolete import utils, parameter_manager as pm, parameter_interpolation as pi

ParameterSpecification = namedtuple('ParameterSpecification', ['id', 'name', 'param_id', 'filter', 'must_be_unique'],
                                    defaults=(None, False))
"""
id must be unique
"""


class InterpolationInTimeManager:
    def __init__(self, enfiles, iagos_df):
        if len(enfiles) < 2:
            raise ValueError(f'There must be at least 2 ENfiles')
        enfiles = sorted(enfiles, key=lambda filename: utils.get_timestamp_for_ENfilename(filename))
        date_times = sorted(utils.get_timestamp_for_ENfilename(filename) for filename in enfiles)
        self._datetime_enfile_list = list(zip(date_times, enfiles))
        self._iagos_df = iagos_df[['lat', 'lon', 'air_press_AC']].dropna().sort_index(kind='mergesort')

    def process_iagos_file(self, params_spec):
        params_spec = list(params_spec)
        params_name = [param_spec.name for param_spec in params_spec]
        datetime_enfile_iter = iter(self._datetime_enfile_list)
        t0, enfile0 = next(datetime_enfile_iter)
        t_pair = deque((t0, ), maxlen=2)
        enfile_pair = deque((enfile0, ), maxlen=2)
        params_pair = deque((None, ), maxlen=2)

        iagos_row_iter = self._iagos_df.itertuples(name='IagosRow')
        iagos_row = next(iagos_row_iter)
        # FIXME: try except for 'next' above
        time_of_measurement = iagos_row.Index

        if time_of_measurement < t0:
            raise ValueError(f'the first ENfile has the timestamp={t0} while the first IAGOS measurement has t={time_of_measurement}')

        pva_rows = []

        for t1, enfile1 in datetime_enfile_iter:
            t_pair.append(t1)
            enfile_pair.append(enfile1)
            params_pair.append(None)
            while time_of_measurement <= t1:
                # prepare parameters
                for i in range(2):
                    if params_pair[i] is None:
                        parameter_manager = pm.ParameterManager(enfile_pair[i])
                        params_pair[i] = [parameter_manager.get_parameter(param_spec.param_id, param_spec.filter,
                                                                          param_spec.must_be_unique)
                                          for param_spec in params_spec]

                # interpolate and append result to the list of series
                pva_row = pd.Series(index=params_name, name=time_of_measurement)
                for param_no in range(len(params_spec)):
                    root_node = pi.InterpolationNode(label=None)
                    for i in range(2):
                        param = params_pair[i][param_no]
                        v = param.get_value_at(iagos_row.lat, iagos_row.lon, iagos_row.air_press_AC)
                        pi.InterpolationNode(label=t_pair[i], value=v, parent=root_node)
                    pva_row[params_spec[param_no].name] = root_node.interpolate((time_of_measurement, ))
                pva_rows.append(pva_row)

                # take next iagos row, if any
                try:
                    iagos_row = next(iagos_row_iter)
                except StopIteration:
                    return pd.DataFrame(pva_rows, columns=params_name)
                time_of_measurement = iagos_row.Index
        raise ValueError(f'not enough ENfiles, the last one has the timestap={t_pair[-1]} while there is a IAGOS row with t={time_of_measurement} to process')
