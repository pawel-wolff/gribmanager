# TODO: this package should be independent of grib_*; try to extract a common interface for any kind of spatial data

from gribmanager import utils, grib_keys as gk, grib_manager as gm, parameter_interpolation as pi
import functools
import inspect
import pandas as pd


class HorizontalParameter:
    def __init__(self, parameter):
        self._parameter = parameter

    def get_value_at(self, lat, lon, pressure=None):
        return self._parameter.get_value_at(lat, lon)

    def another_get_value_at(self, lat, lon, pressure=None):
        return self.get_value_at(lat, lon)


class VerticalParameter:
    def __init__(self, parameter_at_all_levels):
        self._parameter_at_all_levels = sorted(parameter_at_all_levels, key=lambda grib_message: grib_message[gk.LEVEL])
        self._level = [grib_message[gk.LEVEL] for grib_message in self._parameter_at_all_levels]
        self._no_levels = len(self._level)

    def _index_and_pressure_of_sandwiching_levels(self, pressure, lat, lon):
        raise NotImplementedError

    def _pressure_of_all_levels(self, lat, lon):
        raise NotImplementedError

    def get_value_at(self, lat, lon, pressure):
        root_node = pi.InterpolationNode(label=None)
        for i, p in self._index_and_pressure_of_sandwiching_levels(pressure, lat, lon):
            value_at_level = self._parameter_at_all_levels[i].get_value_at(lat, lon)
            pi.InterpolationNode(label=p, value=value_at_level, parent=root_node)
        return root_node.interpolate((pressure, ))

    def get_vertical_profile_at(self, lat, lon):
        pressures = list(self._pressure_of_all_levels(lat, lon))
        return pd.Series(data=[self._parameter_at_all_levels[i].get_value_at(lat, lon) for i in range(len(pressures))],
                         index=pressures)

    def another_get_value_at(self, lat, lon, pressure):
        """ TODO: deprecated (remove)

        :param pressure: pressure in [Pa]
        :param lat: lattitude from the grid
        :param lon: longitute from the grid
        :return: value of the parameter at the given 3d-location
        """
        parameter_at_any_level = self._parameter_at_all_levels[0]
        points = parameter_at_any_level.get_four_nearest_points(lat, lon)
        root_node = pi.InterpolationNode(label=None)
        # FIXME: remove code duplication with grib_manager.GribMessage.get_value_at
        for same_latitude_points in points:
            lat_node = pi.InterpolationNode(label=same_latitude_points[0].lat, parent=root_node)
            for point in same_latitude_points:
                lon_node = pi.InterpolationNode(label=pi.Longitude(point.lon), parent=lat_node)
                for i, p in self._index_and_pressure_of_sandwiching_levels(pressure, point.lat, point.lon):
                # FIXME: it was like this: for i, p in self.index_and_pressure_of_sandwiching_levels(pressure, lat, lon):
                    #value_at_level2 = self._parameter_at_all_levels[i].get_value_at(point.lat, point.lon)
                    value_at_level = self._parameter_at_all_levels[i].get_value_by_index(point.index)
                    #if value_at_level != value_at_level2:
                    #    raise Exception(f'value_at_level={value_at_level} does not match value_at_level2={value_at_level2}')
                    pi.InterpolationNode(label=p, value=value_at_level, parent=lon_node)

        return root_node.interpolate((lat, lon, pressure))


class VerticalParameterInModelLevel(VerticalParameter):
    # TODO: manage failure cases
    def __init__(self, parameter_at_all_levels, surface_pressure):
        """

        :param parameter_at_all_levels: an iterable with GRIB messages corresponding to a given parameter at its all vertical levels
        :param surface_pressure:
        """
        super().__init__(parameter_at_all_levels)
        self._surface_pressure = surface_pressure
        self._a, self._b = self._get_model_level_definition_coefficients()

    def _get_pressure_of_model_level(self, surface_pressure_at_location, index):
        level = self._level[index]
        return self._a[level] + self._b[level] * surface_pressure_at_location

    def _pressure_of_all_levels(self, lat, lon):
        surface_pressure = self._surface_pressure.get_value_at(lat, lon)
        return (self._get_pressure_of_model_level(surface_pressure, i) for i in range(self._no_levels))

    # TODO: can be optimized with binary search; look at profiling
    def _index_and_pressure_of_sandwiching_levels(self, pressure, lat, lon):
        surface_pressure = self._surface_pressure.get_value_at(lat, lon)
        return utils.sandwiching_values_by_binary_search(pressure, 0, self._no_levels-1,
                                                         functools.partial(self._get_pressure_of_model_level, surface_pressure), aux=(lat, lon))

    def _get_model_level_definition_coefficients(self):
        level_definition_coefficients = self._parameter_at_all_levels[0][gk.PV]
        n_levels = len(level_definition_coefficients) // 2 - 1
        a_in_Pa = level_definition_coefficients[:(n_levels + 1)]
        b_coeff = level_definition_coefficients[(n_levels + 1):]
        return a_in_Pa, b_coeff


class VerticalParameterInPressureLevel(VerticalParameter):
    def __init__(self, parameter_at_all_levels):
        super().__init__(parameter_at_all_levels)
        hPa_as_unit = self._parameter_at_all_levels[0][gk.TYPE_OF_LEVEL] == gk.ISOBARIC_IN_HPA_LEVEL_TYPE
        self._pressure_by_level = [100. * level for level in self._level] if hPa_as_unit else self._level

    def _pressure_of_all_levels(self, lat, lon):
        return self._pressure_by_level

    def _index_and_pressure_of_sandwiching_levels(self, pressure, lat, lon):
        return utils.sandwiching_values_by_binary_search(pressure, 0, self._no_levels-1,
                                                         lambda index: self._pressure_by_level[index], aux=(lat, lon))


class ParameterManager:
    _INDEXING_KEYS = [gk.PARAMETER_ID]

    def __init__(self, grib_filename):
        self._grib_file_indexed = gm.GribFileIndexedByWithCache(grib_filename, *ParameterManager._INDEXING_KEYS)

    def get_parameter(self, param_id, predicate=None, must_be_unique=False):
        def arguments_to_string():
            if predicate is not None:
                try:
                    predicate_source = inspect.getsource(predicate)
                except OSError:
                    predicate_source = '<source code not available>'
            else:
                predicate_source = 'None'
            return f'GRIB file={str(self._grib_file_indexed)}, param_id={param_id}, predicate={predicate_source}, must_be_unique={must_be_unique}'

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
                raise ValueError(f'{arguments_to_string()}: not a vertical parameter (neither it is in model level, nor in pressure level) or an unknown vertical parameter')
        elif len(parameters) == 1:
            # horizontal (2d) parameter
            return HorizontalParameter(parameters[0])
        else:
            raise ValueError(f'{arguments_to_string()}: no such parameters found')
