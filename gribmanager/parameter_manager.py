# TODO: this package should be independent of grib_*; try to extract a common interface for any kind of spatial data

import functools
import inspect
import pandas as pd

from . import utils
from . import grib_keys as gk, grib_manager as gm


class Parameter:
    def __repr__(self):
        return repr(type(self))


class HorizontalParameter(Parameter):
    def __init__(self, parameter):
        super().__init__()
        self._parameter = parameter

    def get_value_at(self, lat, lon, pressure=None):
        return self._parameter.get_value_at(lat, lon)

    def __repr__(self):
        dump = [super().__repr__()]
        dump.append(f'2d parameter: {self._parameter.get(gk.SHORT_NAME)} - {self._parameter.get(gk.NAME)} '
                    f'(parameter id={self._parameter.get(gk.PARAMETER_ID)})')
        dump.append(f'\tlevel: {self._parameter.get(gk.LEVEL)}')
        dump.append(f'\ttype of level: {self._parameter.get(gk.TYPE_OF_LEVEL)}')
        return '\n'.join(dump)


class VerticalParameter(Parameter):
    def __init__(self, parameter_at_all_levels):
        super().__init__()
        self._parameter_at_all_levels = sorted(parameter_at_all_levels, key=lambda grib_message: grib_message[gk.LEVEL])
        self._level = [grib_message[gk.LEVEL] for grib_message in self._parameter_at_all_levels]
        self._no_levels = len(self._level)

    def _index_and_pressure_of_sandwiching_levels(self, pressure, lat, lon):
        raise NotImplementedError

    def _pressure_of_all_levels(self, lat, lon):
        raise NotImplementedError

    def get_value_at(self, lat, lon, pressure):
        i_list, p_list = self._index_and_pressure_of_sandwiching_levels(pressure, lat, lon)
        v_list = (self._parameter_at_all_levels[i].get_value_at(lat, lon) for i in i_list)
        if len(i_list) == 1:
            return next(v_list)
        else:
            return utils.linear_interpolation(pressure, zip(p_list, v_list))

    def get_vertical_profile_at(self, lat, lon):
        pressures = list(self._pressure_of_all_levels(lat, lon))
        return pd.Series(data=[self._parameter_at_all_levels[i].get_value_at(lat, lon) for i in range(len(pressures))],
                         index=pressures)

    def __repr__(self):
        dump = [super().__repr__()]
        parameter_at_any_level = self._parameter_at_all_levels[0]
        dump.append(f'3d parameter: {parameter_at_any_level.get(gk.SHORT_NAME)} - {parameter_at_any_level.get(gk.NAME)} '
                    f'(parameter id={parameter_at_any_level.get(gk.PARAMETER_ID)})')
        dump.append(f'\tnumber of available levels: {self._no_levels}')
        dump.append(f'\tavailable levels: {self._level}')
        return '\n'.join(dump)


class VerticalParameterInModelLevel(VerticalParameter):
    # TODO: manage failure cases
    def __init__(self, parameter_at_all_levels, surface_pressure):
        """

        :param parameter_at_all_levels: an iterable with GRIB messages corresponding to a given parameter at its all vertical levels
        :param surface_pressure:
        """
        super().__init__(parameter_at_all_levels)
        self._surface_pressure = surface_pressure
        self._no_model_levels, self._a, self._b = self._get_model_level_definitions()

    def _get_pressure_of_model_level(self, surface_pressure_at_location, index):
        level = self._level[index]
        # self._a[level] + self._b[level] * surface_pressure_at_location
        # is the pressure at a half-level (the interface between level layers)
        # and we need the pressure at the full-level
        return 0.5 * (self._a[level] + self._a[level - 1]) + \
               0.5 * (self._b[level] + self._b[level - 1]) * surface_pressure_at_location

    def _pressure_of_all_levels(self, lat, lon):
        surface_pressure = self._surface_pressure.get_value_at(lat, lon)
        return (self._get_pressure_of_model_level(surface_pressure, i) for i in range(self._no_levels))

    def _index_and_pressure_of_sandwiching_levels(self, pressure, lat, lon):
        surface_pressure = self._surface_pressure.get_value_at(lat, lon)
        return utils.sandwiching_values_by_binary_search(
            pressure, 0, self._no_levels - 1,
            functools.partial(self._get_pressure_of_model_level, surface_pressure),
            aux=(lat, lon)
        )

    def _get_model_level_definitions(self):
        level_definition_coefficients = self._parameter_at_all_levels[0][gk.PV]
        no_model_levels = len(level_definition_coefficients) // 2 - 1
        a_in_Pa = level_definition_coefficients[:(no_model_levels + 1)]
        b_coeff = level_definition_coefficients[(no_model_levels + 1):]
        return no_model_levels, a_in_Pa, b_coeff

    def __repr__(self):
        dump = [super().__repr__()]
        parameter_at_any_level = self._parameter_at_all_levels[0]
        dump.append(f'\ttype of levels: {parameter_at_any_level.get(gk.TYPE_OF_LEVEL)}')
        dump.append(f'\tnumber of model levels: {self._no_model_levels}')
        return '\n'.join(dump)


# TODO: manage case when at some levels are in hPa and others in Pa; must fix super().__init__ in that respect (sorting wrt level)
class VerticalParameterInPressureLevel(VerticalParameter):
    def __init__(self, parameter_at_all_levels):
        super().__init__(parameter_at_all_levels)
        hPa_as_unit = self._parameter_at_all_levels[0][gk.TYPE_OF_LEVEL] == gk.ISOBARIC_IN_HPA_LEVEL_TYPE
        self._pressure_by_level = [100. * level for level in self._level] if hPa_as_unit else self._level

    def _pressure_of_all_levels(self, lat, lon):
        return self._pressure_by_level

    def _index_and_pressure_of_sandwiching_levels(self, pressure, lat, lon):
        return utils.sandwiching_values_by_binary_search(
            pressure, 0, self._no_levels - 1,
            lambda index: self._pressure_by_level[index],
            aux=(lat, lon)
        )

    def __repr__(self):
        dump = [super().__repr__()]
        parameter_at_any_level = self._parameter_at_all_levels[0]
        dump.append(f'\ttype of levels: {parameter_at_any_level.get(gk.TYPE_OF_LEVEL)}')
        return '\n'.join(dump)


class ParameterManager:
    _INDEXING_KEYS = [gk.PARAMETER_ID]

    def __init__(self, grib_filename):
        self._grib_filename = grib_filename
        self._grib_file_indexed = gm.open_grib(self._grib_filename, index_keys=ParameterManager._INDEXING_KEYS)

    def get_parameter(self, param_id, predicate=None, must_be_unique=False):
        def arguments_to_string():
            if predicate is not None:
                try:
                    predicate_source = inspect.getsource(predicate)
                except OSError:
                    predicate_source = '<source code not available>'
            else:
                predicate_source = 'None'
            return f'GRIB file={str(self._grib_file_indexed)}, param_id={param_id}, predicate={predicate_source}, ' \
                   f'must_be_unique={must_be_unique}'

        parameters = self._grib_file_indexed[param_id]
        if predicate is not None:
            parameters = list(filter(predicate, parameters))
        if len(parameters) > 1:
            if must_be_unique:
                raise ValueError(f'{arguments_to_string()}: a parameter is not unique')

            # vertical (3d) parameter
            if all(parameter.is_level_hybrid() for parameter in parameters):
                surface_pressure = self.get_parameter(gk.SURFACE_PRESSURE_PARAM_ID, must_be_unique=True)
                return VerticalParameterInModelLevel(parameters, surface_pressure)
            elif all(parameter.is_level_isobaric() for parameter in parameters):
                return VerticalParameterInPressureLevel(parameters)
            else:
                raise ValueError(f'{arguments_to_string()}: not a vertical parameter (neither it is in model level, '
                                 f'nor in pressure level) or an unknown vertical parameter')
        elif len(parameters) == 1:
            # horizontal (2d) parameter
            return HorizontalParameter(parameters[0])
        else:
            raise ValueError(f'{arguments_to_string()}: no such parameters found')

    def __repr__(self):
        dump = [repr(type(self))]
        dump.append(f'GRIB file {self._grib_filename}')
        return '\n'.join(dump)
