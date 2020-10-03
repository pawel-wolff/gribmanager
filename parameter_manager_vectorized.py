# TODO: this package should be independent of grib_*; try to extract a common interface for any kind of spatial data

import abc
import functools
import inspect
import numpy as np
import pandas as pd
import xarray as xr
import scipy.interpolate
from typing import Iterable, List, Optional
from common.log import logger
from common import longitude, utils, interpolation
from gribmanager import grib_keys as gk, grib_manager as gm


MODEL_LEVEL_DIM = 'ml'
PRESSURE_LEVEL_DIM = 'pl'
LAT_DIM = 'lat'
LON_DIM = 'lon'
SURFACE_PRESSURE_SHORTNAME = 'sp'


def clip_latitude(arr):
    """
    Limits values to the array to the interval [-90., 90.]
    and logs a warning if there is any value outside this interval.

    :param arr: NumPy array (or scalar)
    :return: a clipped NumPy array (or scalar)
    """

    count = np.count_nonzero(arr < -90.) + np.count_nonzero(arr > 90.)
    if count > 0:
        logger().warning(f'{count} latitude(s) outside the interval [-90, 90] found: {arr[(arr < -90.) | (arr > 90.)]}; these values were clipped to [-90, 90]')
        return np.clip(arr, -90., 90.)
    else:
        return arr


def normalize_longitude(arr, smallest_lon_coord):
    return (arr - smallest_lon_coord) % 360. + smallest_lon_coord


def clip_and_log(a_min, a_max, arr):
    count = np.count_nonzero(arr < a_min) + np.count_nonzero(arr > a_max)
    if count > 0:
        arr_np = np.asarray(arr)
        below_min = arr_np[arr_np < a_min]
        above_max = arr_np[arr_np > a_max]
        if below_min.size > 0:
            logger().warning(f'{below_min.size} pressure value(s) below min={a_min}: {pd.Series(below_min.flat).describe()}]')
        if above_max.size > 0:
            logger().warning(f'{above_max.size} pressure value(s) above max={a_max}: {pd.Series(above_max.flat).describe()}]')
        if a_min == -np.inf:
            a_min = None
        if a_max == np.inf:
            a_max = None
        return np.clip(arr, a_min, a_max)
    else:
        return arr


def _force_unique_grib_message_per_level(grib_msgs):
    msg_by_level = {}
    duplicates = 0
    for msg in grib_msgs:
        level = msg[gk.LEVEL]
        if level in msg_by_level:
            duplicates += 1
        msg_by_level[level] = msg
    if duplicates > 0:
        logger().warning(f'found {duplicates} of GRIB messages while forcing unique message per level; no_levels={len(msg_by_level)}; last processed GRIB message={msg}')
    return list(msg_by_level.values())


class Parameter(abc.ABC):
    def __init__(self, grib_msg: gm.GribMessage):
        self.short_name = grib_msg.get(gk.SHORT_NAME)
        self.name = grib_msg.get(gk.NAME)
        self.param_id = grib_msg.get(gk.PARAMETER_ID)
        self.data = None

    # no longer needed since 0.15.1 version of xarray; see:
    # https://github.com/pydata/xarray/pull/3758/commits/7042803da07f06d3877cfa2599fa06685db14a83
    def _interp(self, coords):
        da = self.data
        # explicit dimensions of interpolation is the set of all dimensions on which the interpolation variables,
        # explicitely given in the coords dictionary, depend. E.g. if the pressure depends on the dimensions
        # ('lat', 'lon'), then ('lat', 'lon') are in the set
        explicit_interpolation_dimensions = \
            set().union(*(coord.dims for coord in coords.values() if isinstance(coord, xr.DataArray)))
        # run thru all dimensions of da which are not present as coords keys (these we want implicitely to keep
        # untouched by interpolation, i.e. keep the original coordinates of da for these dimensions);
        # because of peculiarity of xr.DataArray.interp method, we must add these dimensions to the coords dictionary
        # along with the original coordinates
        for implicit_interpolation_dimension in set(da.dims).difference(coords.keys()):
            if implicit_interpolation_dimension in explicit_interpolation_dimensions:
                coords[implicit_interpolation_dimension] = da[implicit_interpolation_dimension]
        return da.interp(coords=coords, method='linear', assume_sorted=True)

    @abc.abstractmethod
    def interp(self, lat=None, lon=None, pressure=None):
        pass

    @abc.abstractmethod
    def interp_numpy(self, lat, lon, pressure=None):
        pass

    def __str__(self):
        return f'{type(self)}:\n\t{self.short_name} - {self.name} (parameter id={self.param_id})'

    def __repr__(self):
        return f'{type(self)}:\n\t{self.short_name} - {self.name} (parameter id={self.param_id})\n\tdata: {self.data}'


class HorizontalParameter(Parameter):
    def __init__(self, grib_msg: gm.GribMessage):
        super().__init__(grib_msg)
        arr, self.lat_coords, self.lon_coords = grib_msg.to_numpy_array()
        self.smallest_lon_coord = self.lon_coords[0]
        self.data = xr.DataArray(arr, coords={LAT_DIM: self.lat_coords, LON_DIM: self.lon_coords}, dims=(LAT_DIM, LON_DIM))
        metadata = grib_msg.get_metadata()
        self.data.attrs.update(metadata)

    def interp(self, lat=None, lon=None, pressure=None):
        coords = {}
        if lat is not None:
            coords[LAT_DIM] = clip_latitude(lat)
        if lon is not None:
            coords[LON_DIM] = normalize_longitude(lon, self.smallest_lon_coord)
        return self.data.interp(coords=coords, method='linear', assume_sorted=True)

    def interp_numpy(self, lat, lon, pressure=None):
        """
        Interpolation along a timeseries-like data: lat, lon, pressure are considered to be dependent on the same
        variable(s), not necessarily 1-dimensional.
        In particular must be of equal length (shape).

        :param lat: float, numpy.ndarray, xarray, an iterable
        :param lon: float, numpy.ndarray, xarray, an iterable
        :param pressure: optional, None; ignored; for homogeneity of the method signature between other subclasses of Parameter class
        :return: numpy.ndarray
        """

        lat = np.asarray(lat)
        lon = np.asarray(lon)
        if lat.shape != lon.shape:
            raise ValueError(f'lat and lon must have the same shape; lat.shape={lat.shape}, lon.shape={lon.shape}')

        points = (self.lat_coords, self.lon_coords)
        xi = np.stack([clip_latitude(lat), normalize_longitude(lon, self.smallest_lon_coord)], axis=-1)
        res = scipy.interpolate.interpn(points, self.data.values, xi, method='linear')
        return res if lat.shape != () else res.squeeze()


class VerticalParameter(Parameter):
    def __init__(self, grib_msgs_at_all_levels: List[gm.GribMessage], level_coords, level_dim):
        super().__init__(grib_msgs_at_all_levels[0])
        data_lat_lon_list = [msg.to_numpy_array() for msg in grib_msgs_at_all_levels]
        data_list, lat_coords_list, lon_coords_list = zip(*data_lat_lon_list)
        lat_coords_stacked = np.stack(lat_coords_list)
        self.lat_coords = lat_coords_list[0]
        if not (lat_coords_stacked == self.lat_coords).all():
            raise ValueError(f'latitude coordinates are not coherent across levels; self={self}')
        lon_coords_stacked = np.stack(lon_coords_list)
        self.lon_coords = lon_coords_list[0]
        if not (lon_coords_stacked == self.lon_coords).all():
            raise ValueError(f'longitude coordinates are not coherent across levels; self={self}')
        self.smallest_lon_coord = self.lon_coords[0]
        data_stacked = np.stack(data_list)
        self.data = xr.DataArray(data_stacked,
                                 coords={level_dim: level_coords, LAT_DIM: self.lat_coords, LON_DIM: self.lon_coords},
                                 dims=(level_dim, LAT_DIM, LON_DIM)).sortby(level_dim)
        metadata = grib_msgs_at_all_levels[0].get_metadata()
        self.data.attrs.update(metadata)


class VerticalParameterInModelLevel(VerticalParameter):
    """
    Vertical parameter with eta vertical coordinate (a.k.a. hybrid sigma vertical coordinate, c.f.
    https://rda.ucar.edu/datasets/ds627.0/docs/Eta_coordinate/
    or
    https://confluence.ecmwf.int/display/OIFS/4.4+OpenIFS%3A+Vertical+Resolution+and+Configuration
    or
    https://www.ecmwf.int/en/elibrary/16647-part-iii-dynamics-and-numerical-procedures
    (Part III, Chapter 2)
    """
    # TODO: manage failure cases
    def __init__(self,
                 grib_msgs_at_all_levels: Iterable[gm.GribMessage],
                 surface_pressure: Optional[HorizontalParameter] = None,
                 parameter_on_half_levels=None):
        # TODO: test if this works properly; permute grib_msgs_at_all_levels, a see if the ndarray self.values.values has changed accordingly
        """

        :param grib_msgs_at_all_levels: an iterable with GRIB messages corresponding to a given parameter at its all vertical levels
        :param surface_pressure: optional; HorizontalParameter containing surface pressure
        :param parameter_on_half_levels: optional; if True, the parameter is considered to be given at half-levels of eta vertical coordinate
        """
        grib_msgs_at_all_levels = _force_unique_grib_message_per_level(grib_msgs_at_all_levels)
        self._surface_pressure = None
        self.set_surface_pressure(surface_pressure)
        self.no_levels = len(grib_msgs_at_all_levels)
        if self.no_levels == 0:
            raise ValueError(f'grib_msgs_at_all_levels={grib_msgs_at_all_levels}')
        self.ml_coords, self.a_in_Pa, self.b_coeff = self._get_model_level_coords_and_coeffs(grib_msgs_at_all_levels, parameter_on_half_levels)
        super().__init__(grib_msgs_at_all_levels, level_coords=self.ml_coords, level_dim=MODEL_LEVEL_DIM)

    def set_surface_pressure(self, surface_pressure: HorizontalParameter):
        self._surface_pressure = surface_pressure

    def _get_model_level_coords_and_coeffs(self, grib_msgs_at_all_levels, parameter_on_half_levels):
        type_of_level, paramater_level_index, ab_list \
            = zip(*((msg[gk.TYPE_OF_LEVEL],
                     msg[gk.LEVEL],
                     np.asarray(msg[gk.PV]))
                    for msg in grib_msgs_at_all_levels))
        if not all(tl == gk.HYBRID_LEVEL_TYPE for tl in type_of_level):
            d = {l: tl for l, tl in zip(paramater_level_index, type_of_level) if tl != gk.HYBRID_LEVEL_TYPE}
            raise ValueError(f'Some levels are not {gk.HYBRID_LEVEL_TYPE}: {d}')

        parameter_level_coords = np.asarray(paramater_level_index)
        if parameter_on_half_levels:
            # add 0.5 to parameter_level_coords if the parameter is given at half-levels (or -0.5 ???)
            parameter_level_coords = parameter_level_coords + 0.5
        paramater_level_index_with_guard = paramater_level_index + (max(paramater_level_index) + 1, )
        parameter_level_coords_with_guard = np.asarray(paramater_level_index_with_guard)
        ab_stacked = np.stack(ab_list)
        ab = ab_list[0]
        if not (ab_stacked == ab).all():
            raise ValueError(f'model level definition coefficients (PV: a, b) are not coherent across levels; self={self}')
        a_len = len(ab) // 2
        model_half_level_coords = np.arange(a_len) + 0.5
        a_in_Pa = xr.DataArray(np.array(ab[:a_len]),
                               coords={MODEL_LEVEL_DIM: model_half_level_coords}, dims=MODEL_LEVEL_DIM)
        b_coeff = xr.DataArray(np.array(ab[a_len:]),
                               coords={MODEL_LEVEL_DIM: model_half_level_coords}, dims=MODEL_LEVEL_DIM)
        return parameter_level_coords, \
               a_in_Pa.interp(coords={MODEL_LEVEL_DIM: parameter_level_coords_with_guard},
                              method='linear', assume_sorted=True, kwargs={'bounds_error': False, 'fill_value': np.inf}), \
               b_coeff.interp(coords={MODEL_LEVEL_DIM: parameter_level_coords_with_guard},
                              method='linear', assume_sorted=True, kwargs={'bounds_error': False, 'fill_value': 0.})

    def interp(self, lat=None, lon=None, pressure=None):
        if pressure is None:
            ml = None
        else:
            if self._surface_pressure is None:
                raise ValueError(f'{self} does not know a surface pressure')
            sp = self._surface_pressure.interp(lat=lat, lon=lon)
            p = sp * self.b_coeff + self.a_in_Pa
            # p(t, ml) is increasing in ml; find interpolated ml(t) such that p(t, ml(t)) = p_0(t) for all t
            if not isinstance(pressure, xr.DataArray):
                pressure = xr.DataArray(pressure)
            level_index = (p >= pressure).argmax(dim=MODEL_LEVEL_DIM)
            lower_level_index = np.maximum(level_index - 1, 0)
            upper_level_index = np.minimum(level_index, self.no_levels - 1)
            pressure_lower = p.isel({MODEL_LEVEL_DIM: lower_level_index}).drop(labels=MODEL_LEVEL_DIM)
            pressure_upper = p.isel({MODEL_LEVEL_DIM: upper_level_index}).drop(labels=MODEL_LEVEL_DIM)
            weight = xr.where(abs(pressure_upper - pressure_lower) < 1e-05,
                              0.5,
                              (pressure - pressure_lower) / (pressure_upper - pressure_lower))
            ml = (1 - weight) * self.ml_coords[lower_level_index] + weight * self.ml_coords[upper_level_index]

        coords = {}
        if ml is not None:
            coords[MODEL_LEVEL_DIM] = ml
        if lat is not None:
            coords[LAT_DIM] = clip_latitude(lat)
        if lon is not None:
            coords[LON_DIM] = normalize_longitude(lon, self.smallest_lon_coord)
        return self.data.interp(coords=coords, method='linear', assume_sorted=True)

    def interp_numpy(self, lat, lon, pressure):
        """
        Interpolation along a timeseries-like data: lat, lon, pressure are considered to be dependent on the same
        variable(s), not necessarily 1-dimensional.
        In particular must be of equal length (shape).

        :param lat: float, numpy.ndarray, xarray, an iterable
        :param lon: float, numpy.ndarray, xarray, an iterable
        :param pressure: float, numpy.ndarray, xarray, an iterable
        :return: numpy.ndarray
        """

        pressure = np.asarray(pressure)
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        if not (pressure.shape == lat.shape == lon.shape):
            raise ValueError(f'lat, lon and pressure must have the same shape; lat.shape={lat.shape}, lon.shape={lon.shape}, pressure.shape={pressure.shape}')
        if self._surface_pressure is None:
            raise ValueError(f'{self} does not know a surface pressure')
        sp = self._surface_pressure.interp_numpy(lat=lat, lon=lon)
        p = np.multiply.outer(sp, self.b_coeff.values) + self.a_in_Pa.values
        # p(t, x) is increasing in x; find interpolated x(t) st p(t, x(t)) = p_0(t) for all t
        level_index = np.argmax(p >= pressure[..., np.newaxis], axis=-1)
        lower_level_index = np.maximum(level_index - 1, 0)
        upper_level_index = np.minimum(level_index, self.no_levels - 1)
        pressure_indices = tuple(np.indices(pressure.shape)) # if pressure is d-dim ndarray, then pressure_indices is a list od d d-dim ndarrays with indicies along corresponding dimension
        pressure_lower = p[pressure_indices + (lower_level_index, )]
        pressure_upper = p[pressure_indices + (upper_level_index, )] # TODO: maybe with np.take one can make it better?
        pressure_upper_minus_pressure_lower = pressure_upper - pressure_lower
        pressure_upper_close_to_pressure_lower = np.isclose(pressure_upper_minus_pressure_lower, 0.)
        pressure_minus_pressure_lower_bis = np.where(pressure_upper_close_to_pressure_lower, 1., pressure - pressure_lower)
        pressure_upper_minus_pressure_lower_bis = np.where(pressure_upper_close_to_pressure_lower, 2., pressure_upper_minus_pressure_lower)
        weight = pressure_minus_pressure_lower_bis / pressure_upper_minus_pressure_lower_bis
        ml = (1 - weight) * self.ml_coords[lower_level_index] + weight * self.ml_coords[upper_level_index]

        points = (self.ml_coords, self.lat_coords, self.lon_coords)
        xi = np.stack([ml, clip_latitude(lat), normalize_longitude(lon, self.smallest_lon_coord)], axis=-1)
        return scipy.interpolate.interpn(points, self.data.values, xi, method='linear')


# TODO: manage case when at some levels are in hPa and others in Pa; must fix super().__init__ in that respect (sorting wrt level)
class VerticalParameterInPressureLevel(VerticalParameter):
    def __init__(self, grib_msgs_at_all_levels: Iterable[gm.GribMessage]):
        # TODO: test if this works properly; permute grib_msgs_at_all_levels, a see if the ndarray self.values.values has changed accordingly
        """

        :param grib_msgs_at_all_levels: an iterable with GRIB messages corresponding to a given parameter at its all vertical levels
        """
        grib_msgs_at_all_levels = _force_unique_grib_message_per_level(grib_msgs_at_all_levels)
        self.no_levels = len(grib_msgs_at_all_levels)
        if self.no_levels == 0:
            raise ValueError(f'grib_msgs_at_all_levels={grib_msgs_at_all_levels}')
        level, type_of_level = zip(*((msg[gk.LEVEL], msg[gk.TYPE_OF_LEVEL]) for msg in grib_msgs_at_all_levels))
        isobaric_level_types = (gk.ISOBARIC_IN_PA_LEVEL_TYPE, gk.ISOBARIC_IN_HPA_LEVEL_TYPE)
        if not all(tl in isobaric_level_types for tl in type_of_level):
            d = {l: tl for l, tl in zip(level, type_of_level) if tl not in isobaric_level_types}
            raise ValueError(f'Some levels are not {isobaric_level_types}: {d}')
        self.pl_coords = np.asarray([100. * float(l) if tl == gk.ISOBARIC_IN_HPA_LEVEL_TYPE else float(l) for l, tl in zip(level, type_of_level)])
        self.clip_pressures = functools.partial(clip_and_log, self.pl_coords.min(), self.pl_coords.max())
        super().__init__(grib_msgs_at_all_levels, level_coords=self.pl_coords, level_dim=PRESSURE_LEVEL_DIM)

    def interp(self, lat=None, lon=None, pressure=None):
        coords = {}
        if pressure is not None:
            coords[PRESSURE_LEVEL_DIM] = self.clip_pressures(pressure)
        if lat is not None:
            coords[LAT_DIM] = clip_latitude(lat)
        if lon is not None:
            coords[LON_DIM] = normalize_longitude(lon, self.smallest_lon_coord)
        return self.data.interp(coords=coords, method='linear', assume_sorted=True)

    def interp_numpy(self, lat, lon, pressure):
        """
        Interpolation along a timeseries-like data: lat, lon, pressure (in [Pa]) are considered to be dependent on
        the same variable(s), not necessarily 1-dimensional. In particular must be of equal length (shape).

        :param lat: float, numpy.ndarray, xarray, an iterable
        :param lon: float, numpy.ndarray, xarray, an iterable
        :param pressure: float, numpy.ndarray, xarray, an iterable
        :return: numpy.ndarray
        """

        pressure = np.asarray(pressure)
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        if not (pressure.shape == lat.shape == lon.shape):
            raise ValueError(f'lat, lon and pressure must have the same shape; lat.shape={lat.shape}, lon.shape={lon.shape}, pressure.shape={pressure.shape}')

        points = (self.pl_coords, self.lat_coords, self.lon_coords)
        xi = np.stack([self.clip_pressures(pressure), clip_latitude(lat), normalize_longitude(lon, self.smallest_lon_coord)], axis=-1)
        return scipy.interpolate.interpn(points, self.data.values, xi, method='linear')


# TODO: remove as deprecated
class ParameterManager:
    """
    deprecated
    """
    _INDEXING_KEYS = [gk.PARAMETER_ID]

    def __init__(self, grib_filename):
        self._grib_filename = grib_filename
        self._grib_file_indexed = gm.open_grib(self._grib_filename, index_keys=ParameterManager._INDEXING_KEYS)

    def close(self):
        self._grib_file_indexed.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()

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

        grib_msgs = self._grib_file_indexed[param_id]
        if predicate is not None:
            grib_msgs = list(filter(predicate, grib_msgs))
        if len(grib_msgs) > 1:
            if must_be_unique:
                raise ValueError(f'{arguments_to_string()}: a parameter is not unique')

            # vertical (3d) parameter
            if all(msg.is_level_hybrid() for msg in grib_msgs):
                surface_pressure = self.get_parameter(gk.SURFACE_PRESSURE_PARAM_ID, must_be_unique=True)
                return VerticalParameterInModelLevel(grib_msgs, surface_pressure)
            elif all(msg.is_level_isobaric() for msg in grib_msgs):
                return VerticalParameterInPressureLevel(grib_msgs)
            else:
                raise ValueError(f'{arguments_to_string()}: not a vertical parameter (neither it is in model level, '
                                 f'nor in pressure level) or an unknown vertical parameter')
        elif len(grib_msgs) == 1:
            # horizontal (2d) parameter
            return HorizontalParameter(grib_msgs[0])
        else:
            raise ValueError(f'{arguments_to_string()}: no such parameters found')

    def __repr__(self):
        dump = [repr(type(self))]
        dump.append(f'GRIB file {self._grib_filename}')
        return '\n'.join(dump)


_RESERVED_PARAM_SPEC_KEYS = ['name', 'param_id', 'must_be_unique']


def load_grib_parameters(filenames, params_spec, ignore_not_found=False, surface_pressure=None, use_eccodes_index=False):
    """
    Load ECMWF parameters contained in a single or multiple GRIB files

    :param filenames: a path to a GRIB file or, in the case of multiple GIRB files, a list or a tuple of filenames.
    In case of conflicting parameters read from several files, the parameter from the latest file is kept.
    :param params_spec: a list of dictionaries; each dictionary must specify an ECMWF parameter
    to be loaded from the GRIB file. The dictionary must have the following keys and values:
    key: 'name', value: str; use 'sp' for a surface pressure parameter as it will be used to interpolate parameters
    in model level (unless a surface pressure parameter is explicitly passed via 'surface_pressure' parameter)
    key: 'param_id', value: int, ECMWF parameter id
    key: 'must_be_unique', value: bool; indicates whether the parameter is expected to be represented by a single GRIB message
    Furthermore, the following keys are optional:
    key: any valid GRIB key, value: a single value or a list of values of the GRIB key to be used as a filter of GRIB messages.
    In case of conflicting parameters read from the same file via several different specification entries, the parameter
    corresponding to the earliest specification on the list is kept.
    :param ignore_not_found: optional, default False; if True then ignores parameters which cannot be found or loaded successfully
    :param surface_pressure, value: HorizontalParameter; a surfrace pressure parameter if it is known and None otherwise
    :param use_eccodes_index: optional, default False; if True then uses an ecCodes' indexing feature
    (not recommended because of apparent file descriptor leak - 28/09/2020)
    :return: a dict of Parameter objects, with keys being names given in params_spec
    """
    if not isinstance(filenames, (list, tuple)):
        filenames = (filenames, )
    params_global_dict = {}
    for filename in filenames:
        if not params_spec:
            break
        if use_eccodes_index:
            params_current_dict = _load_grib_parameters_from_single_file_using_index(filename, params_spec, surface_pressure=surface_pressure)
        else:
            params_current_dict = _load_grib_parameters_from_single_file(filename, params_spec, surface_pressure=surface_pressure)
        params_spec = [param_spec for param_spec in params_spec if param_spec['name'] not in params_current_dict]
        if not surface_pressure and SURFACE_PRESSURE_SHORTNAME in params_current_dict:
            surface_pressure = params_current_dict[SURFACE_PRESSURE_SHORTNAME]
        params_global_dict.update(params_current_dict)
    if params_spec and not ignore_not_found:
        raise ValueError(f'In GRIB file(s) {filenames} {len(params_spec)} parameters were not found: {params_spec}')
    return params_global_dict


def _load_grib_parameters_from_single_file(filename, params_spec, surface_pressure=None):
    def get_param_from_msgs(msgs, param_spec):
        if param_spec['must_be_unique'] and len(msgs) > 1:
            logger().warning(
                f'{len(msgs)} GRIB messages were found in the GRIB file {filename} with param_spec={param_spec} '
                f'while only one was expected; taking the last GRIB message')
            msgs = msgs[-1:]

        if len(msgs) > 1:
            # vertical (3d) parameter
            if all(msg.is_level_hybrid() for msg in msgs):
                return VerticalParameterInModelLevel(msgs, surface_pressure=surface_pressure)
            elif all(msg.is_level_isobaric() for msg in msgs):
                return VerticalParameterInPressureLevel(msgs)
            else:
                raise ValueError(
                    f'messages in GRIB file {filename} filtered according to param_spec={param_spec} '
                    f'does not form any known vertical parameter (neither it is in model level nor in pressure level); '
                    f'number of filtered messages={len(msgs)}')
        else:
            # len(grib_msgs) == 1
            # horizontal (2d) parameter
            return HorizontalParameter(msgs[0])

    open_msgs = []
    try:
        # here we build a dictionary (param_name, list of GRIB messages)
        msgs_by_param_name = {}
        spec_priority_by_param_name = {}
        group_of_priority_and_param_spec_by_param_id = \
            utils.groupby(enumerate(params_spec),
                          lambda priority_and_param_spec: priority_and_param_spec[1]['param_id'])

        with gm.open_grib(filename) as grib:
            for msg in grib:
                try:
                    keep_msg_open = False
                    try:
                        param_id = msg[gk.PARAMETER_ID]
                    except KeyError:
                        logger().warning(
                            f'grib message {msg} in the GRIB file {filename} does not have the GRIB key={gk.PARAMETER_ID}; '
                            f'the message is ignored'
                        )
                        msg.close()
                        continue
                    try:
                        group_of_priority_and_param_spec = group_of_priority_and_param_spec_by_param_id[param_id]
                    except KeyError:
                        msg.close()
                        continue
                    for spec_priority, param_spec in group_of_priority_and_param_spec:
                        filter_on = [(key, value) for key, value in param_spec.items() if
                                     key not in _RESERVED_PARAM_SPEC_KEYS]
                        cond = True
                        for key, value in filter_on:
                            try:
                                v = msg[key]
                            except KeyError:
                                logger().warning(
                                    f'grib message {msg} in the GRIB file {filename} does not have the GRIB key={key} '
                                    f'on which it was supposed to be filtered; value={value}; the message is ignored')
                                cond = False
                                break
                            if isinstance(value, (list, tuple)):
                                cond = cond and v in value
                            else:
                                cond = cond and v == value
                        if cond:
                            param_name = param_spec['name']
                            previous_spec_priority = spec_priority_by_param_name.get(param_name, default=None)
                            if previous_spec_priority is None or previous_spec_priority > spec_priority:
                                spec_priority_by_param_name[param_name] = spec_priority
                                if previous_spec_priority is not None:
                                    del spec_priority_by_param_name[param_name]
                                msgs_by_param_name.setdefault(param_name, []).append(msg)
                                keep_msg_open = True

                    if keep_msg_open:
                        open_msgs.append(msg)
                    else:
                        msg.close()
                except Exception as e:
                    msg.close()
                    raise e

        # we build Parameters from GIRB messages
        param_by_param_name = {}
        for param_spec in params_spec:
            param_name = param_spec['name']
            try:
                msgs = msgs_by_param_name[param_name]
            except KeyError:
                continue
            try:
                param = get_param_from_msgs(msgs, param_spec)
            except Exception as e:
                logger().exception(f'LOAD_GRIB_PAREMETERS_ERROR: cannot load ECMWF parameter from the GRIB file={filename} '
                                 f'with param_spec={param_spec}', exc_info=e)
                continue
            if param is not None:
                param_by_param_name[param_name] = param

        # complete eventual VerticalParameterInModelLevel's with surface pressure parameter
        # if that one was read in the GRIB file
        if surface_pressure is None and SURFACE_PRESSURE_SHORTNAME in param_by_param_name:
            just_read_surface_pressure = param_by_param_name[SURFACE_PRESSURE_SHORTNAME]
            if isinstance(just_read_surface_pressure, HorizontalParameter):
                for param in param_by_param_name.values():
                    if isinstance(param, VerticalParameterInModelLevel):
                        param.set_surface_pressure(just_read_surface_pressure)
            else:
                surface_pressure_spec = utils.unique(param_spec for param_spec in params_spec if param_spec['name'] == SURFACE_PRESSURE_SHORTNAME)
                logger().warning(f'a surface pressure read from the GRIB file={filename} '
                               f'using param_spec={surface_pressure_spec} is not a HorizontalParameter; '
                               f'it is: {just_read_surface_pressure}')
        return param_by_param_name
    finally:
        # whatever happens (normal return or an exception), we do a cleanup
        for msg in open_msgs:
            msg.close()


def _load_grib_parameters_from_single_file_using_index(filename, params_spec, surface_pressure=None):
    def get_param_by_id(param_id, must_be_unique, filter_on=None):
        nonlocal surface_pressure
        grib_msgs = grib.get(param_id, default=[])
        try:
            logger().debug(f'param_id={param_id}, len(grib_msgs)={len(grib_msgs)}, filter_on={filter_on}')
            if filter_on:
                filtered_grib_msgs = []
                for msg in grib_msgs:
                    cond = True
                    for key, value in filter_on:
                        try:
                            v = msg[key]
                        except KeyError:
                            logger().warning(f'grib message {msg} in the GRIB file {filename} does not have the GRIB key={key} on which it was supposed to be filtered; value={value}')
                            cond = False
                            break
                        if isinstance(value, (list, tuple)):
                            cond = cond and v in value
                        else:
                            cond = cond and v == value
                    if cond:
                        filtered_grib_msgs.append(msg)
            else:
                filtered_grib_msgs = grib_msgs

            if not filtered_grib_msgs:
                return None

            if must_be_unique and len(filtered_grib_msgs) > 1:
                logger().warning(f'more than one grib message found in the GRIB file {filename} with param_id={param_id} and filter_on={filter_on}, '
                               f'while only one was expected; taking the last grib message; number of filtered messages={len(filtered_grib_msgs)}')
                filtered_grib_msgs = filtered_grib_msgs[-1:]

            if len(filtered_grib_msgs) > 1:
                # vertical (3d) parameter
                if all(msg.is_level_hybrid() for msg in filtered_grib_msgs):
                    if not surface_pressure:
                        surface_pressure = get_param_by_id(gk.SURFACE_PRESSURE_PARAM_ID, must_be_unique=True)
                    return VerticalParameterInModelLevel(filtered_grib_msgs, surface_pressure)
                elif all(msg.is_level_isobaric() for msg in filtered_grib_msgs):
                    return VerticalParameterInPressureLevel(filtered_grib_msgs)
                else:
                    raise ValueError(f'messages in GRIB file {filename} filtered according the criteria: param_id={param_id}, must_be_unique={must_be_unique}, filter_on={filter_on} '
                                     f'does not form any known vertical parameter (neither it is in model level nor in pressure level); number of filtered messages={len(filtered_grib_msgs)}')
            else:
                # len(grib_msgs) == 1
                # horizontal (2d) parameter
                return HorizontalParameter(filtered_grib_msgs[0])
        finally:
            for msg in grib_msgs:
                msg.close()

    params = {}
    with gm.open_grib(filename, index_keys=gk.PARAMETER_ID) as grib:
        for param_spec in params_spec:
            name = param_spec['name']
            param_id = param_spec['param_id']
            must_be_unique = param_spec['must_be_unique']
            filter_on = [(key, value) for key, value in param_spec.items() if key not in _RESERVED_PARAM_SPEC_KEYS]
            try:
                param = get_param_by_id(param_id, must_be_unique, filter_on)
            except Exception as e:
                logger().exception(f'LOAD_GRIB_PAREMETERS_ERROR: cannot load ECMWF parameter from the GRIB file={filename} with name={name}, '
                                 f'param_id={param_id}, must_be_unique={must_be_unique}, filter_on={filter_on}', exc_info=e)
                continue
            if param is not None:
                params[name] = param
    return params
