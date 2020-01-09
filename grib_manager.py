# TODO: look into pygrid library: https://github.com/jswhit/pygrib/blob/master/pygrib.pyx; https://jswhit.github.io/pygrib/docs/pygrib-module.html

import functools
import math
import eccodes as ecc
from common.log import logger
from common import utils, abstract_dictionary, longitude, interpolation
from gribmanager import grib_keys as gk


_grib_items = 0
_grib_messages = 0
_grib_messages_released = 0
_grib_files = 0
_grib_files_closed = 0
_grib_indices = 0
_grib_indices_released = 0


_GRIB_KEY_NAMESPACE_USED_FOR_PRINT_GRIB_MESSAGE = 'mars'
_GRIB_KEY_NAMESPACE = 'parameter'
_GRIB_KEY_NAMESPACE_USED_KEY_ITERATION = None


class GribAbstractItem:
    def __init__(self):
        self._id = None
        global _grib_items
        _grib_items += 1

    def __enter__(self):
        return self

    # FIXME: how to do it properly? see e.g.
    # https://stackoverflow.com/questions/32975036/why-is-del-called-at-the-end-of-a-with-block
    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.__del__()
        pass


# basic implementation of access to a GRIB message
class GribMessage(abstract_dictionary.AbstractDictionary, GribAbstractItem):
    def __init__(self, message_id, grib_file):
        super().__init__()
        global _grib_messages
        _grib_messages += 1
        self._id = message_id
        self._grib_file = grib_file
        self._values = None
        self._get_lat_lon_index_of_four_nearest_points = self._check_grid()
        logger.info(f'GribMessage init id={self.get_id()}')

    def _check_grid(self):
        if gk.VALUES in self and self.get(gk.PACKING_TYPE) == gk.PACKING_TYPE_GRID_SIMPLE \
                and self.get(gk.GRID_TYPE) == gk.GRID_TYPE_REGULAR_LL:
            lat0, lon0 = self[gk.LATITUDE_OF_FIRST_GRID_POINT], self[gk.LONGITUDE_OF_FIRST_GRID_POINT]
            lat1, lon1 = self[gk.LATITUDE_OF_LAST_GRID_POINT], self[gk.LONGITUDE_OF_LAST_GRID_POINT]
            d_lat = abs(self[gk.DELTA_LATITUDE]) * (1 if self[gk.DELTA_LATITUDE_POSITIVE] else -1)
            d_lon = abs(self[gk.DELTA_LONGITUDE]) * (-1 if self[gk.DELTA_LONGITUDE_NEGATIVE] else 1)
            lat_major = False if self[gk.LATITUDE_MINOR_LONGITUDE_MAJOR] else True
            n_lat, n_lon = self[gk.NO_LATITUDES], self[gk.NO_LONGITUDES]
            if n_lat <= 1 or n_lon <= 1:
                raise ValueError
            if not math.isclose(lat0 + (n_lat - 1) * d_lat, lat1):
                raise ValueError
            # TODO: similar check for longitudes, but be careful
            return functools.partial(utils.four_nearest_points_in_rectangular_grid, lat0, lon0, lat1, lon1, d_lat, d_lon, n_lat, n_lon, lat_major)
        else:
            return None

    def get_id(self):
        if self._id is None:
            raise ValueError('GRIB message already released')
        return self._id

    def __del__(self):
        if self._id is not None:
            logger.info(f'releasing GribMessage id={self.get_id()}')
            ecc.codes_release(self.get_id())
            self._id = None
            global _grib_messages_released
            _grib_messages_released += 1

    def __contains__(self, key):
        if ecc.codes_is_defined(self.get_id(), key):
            return True
        return False

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(f'GRIB message does not contain key={key}: {str(self)}')
        if ecc.codes_is_missing(self.get_id(), key):
            logger.warning(f'key={key} is has value=MISSING in the GRIB message: {str(self)}')
        if ecc.codes_get_size(self.get_id(), key) > 1:
            return ecc.codes_get_array(self.get_id(), key)
        else:
            return ecc.codes_get(self.get_id(), key)

    def get_four_nearest_points(self, lat, lon, check_assertion=False, use_eccodes_routine=False):
        if not use_eccodes_routine and self._get_lat_lon_index_of_four_nearest_points is not None:
            points = self._get_lat_lon_index_of_four_nearest_points(lat, lon)
            if check_assertion:
                # TODO: remove the assertion
                indices_eccodes = {point.index for point in self.get_four_nearest_points(lat, lon, use_eccodes_routine=True)}
                indices = {point.index for point in points}
                assert indices == indices_eccodes, (lat, lon, indices, indices_eccodes)
            return points
        else:
            points = sorted(ecc.codes_grib_find_nearest(self.get_id(), lat, lon, npoints=4), key=lambda point: point.lat)
            return (points[0], points[1]), (points[2], points[3])

    def get_value_at(self, lat, lon):
        root_node = interpolation.InterpolationNode(label=None)
        points = self.get_four_nearest_points(lat, lon)
        for same_latitude_points in points:
            lat_node = interpolation.InterpolationNode(label=same_latitude_points[0].lat, parent=root_node)
            for point in same_latitude_points:
                interpolation.InterpolationNode(label=longitude.Longitude(point.lon), value=self.get_value_by_index(point.index), parent=lat_node)
        return root_node.interpolate((lat, longitude.Longitude(lon)))

    def get_value_by_index(self, index):
        if self._values is None:
            self._values = self[gk.VALUES]
        return self._values[index]

    def is_level_hybrid(self):
        return gk.LEVEL in self and self.get(gk.TYPE_OF_LEVEL) == gk.HYBRID_LEVEL_TYPE and gk.PV in self

    def is_level_isobaric(self):
        return gk.LEVEL in self and self.get(gk.TYPE_OF_LEVEL) in [gk.ISOBARIC_IN_HPA_LEVEL_TYPE,
                                                                   gk.ISOBARIC_IN_PA_LEVEL_TYPE]

    def is_level_surface(self):
        return self.get(gk.TYPE_OF_LEVEL) == gk.SURFACE_LEVEL

    def __str__(self):
        output = f'filename={str(self._grib_file)}'
        for key_name in _GribMessageKeyIterator(self, key_namespace=_GRIB_KEY_NAMESPACE_USED_FOR_PRINT_GRIB_MESSAGE):
            output += f', {key_name}={self.get(key_name)}'
        return output

    def __repr__(self):
        dump = [repr(type(self))]
        dump.append(f'GRIB message; fast grided data access: '
                      f'{"yes" if self._get_lat_lon_index_of_four_nearest_points is not None else "no"}')
        dump.append(f'\tOriginating generating centre: {self.get(gk.CENTRE)}')
        dump.append(f'\tReference date: {self.get(gk.REFERENCE_DATE)}, time: {self.get(gk.REFERENCE_TIME)}')
        dump.append(f'\tParameter id: {self.get(gk.PARAMETER_ID)}')
        dump.append(f'\tShort name: {self.get(gk.SHORT_NAME)}')
        dump.append(f'\tName: {self.get(gk.NAME)}')
        dump.append(f'\tUnits: {self.get(gk.UNITS)}')
        dump.append(f'\tType of level: {self.get(gk.TYPE_OF_LEVEL)}')
        dump.append(f'\tLevel: {self.get(gk.LEVEL)}')
        grid_type = self.get(gk.GRID_TYPE)
        dump.append(f'\tGrid type: {grid_type}')
        if grid_type == gk.GRID_TYPE_REGULAR_LL:
            dump.append(f'\t\tGridded area (lat, lon): '
                          f'({self.get(gk.LATITUDE_OF_FIRST_GRID_POINT)}, {self.get(gk.LONGITUDE_OF_FIRST_GRID_POINT)}), '
                          f'({self.get(gk.LATITUDE_OF_LAST_GRID_POINT)}, {self.get(gk.LONGITUDE_OF_LAST_GRID_POINT)})')
            dump.append(f'\tGrid resolution: '
                          f'd_lat={self.get(gk.DELTA_LATITUDE)}, d_lon={self.get(gk.DELTA_LONGITUDE)}')
        elif grid_type == gk.GRID_TYPE_SH:
            dump.append(f'\t\tM={self.get(gk.GRID_SH_M)}, K={self.get(gk.GRID_SH_K)}, J={self.get(gk.GRID_SH_J)}')
        return '\n'.join(dump)

    def __iter__(self):
        return _GribMessageKeyIterator(self)


# lazy dictionary-like implementation of access to a GRIB message
class GribMessageWithCache(abstract_dictionary.AbstractCacheDictionary, GribMessage):
    pass


class _GribMessageKeyIterator:
    def __init__(self, grib_message, key_namespace=_GRIB_KEY_NAMESPACE_USED_KEY_ITERATION):
        self._grib_message = grib_message
        self._keys_iterator_id = \
            ecc.codes_keys_iterator_new(self._grib_message.get_id(), key_namespace)
        ecc.codes_skip_duplicates(self._keys_iterator_id)
        #ecc.codes_skip_computed(self._keys_iterator_id)
        #ecc.codes_skip_edition_specific(self._keys_iterator_id)
        #ecc.codes_skip_function(self._keys_iterator_id)

    def __del__(self):
        ecc.codes_keys_iterator_delete(self._keys_iterator_id)

    def __iter__(self):
        return self

    def __next__(self):
        if ecc.codes_keys_iterator_next(self._keys_iterator_id):
            return ecc.codes_keys_iterator_get_name(self._keys_iterator_id)
        else:
            raise StopIteration


def open_grib(filename, index_keys=None, unique_indexing=False):
    if index_keys is None:
        return GribFile(filename)
    elif not unique_indexing:
        return GribFileIndexedByWithCache(filename, *index_keys)
    else:
        return GribFileUniquelyIndexedByWithCache(filename, *index_keys)


class GribFile(GribAbstractItem):
    def __init__(self, filename):
        self._file = None
        super().__init__()
        self._filename = filename
        self._file = open(filename, 'rb')
        global _grib_files
        _grib_files += 1

    def get_file(self):
        if self._file.closed:
            raise ValueError(f'GRIB file {str(self)} already closed')
        return self._file

    def __del__(self):
        if self._file is not None and not self._file.closed:
            logger.info(f'releasing GribFile {str(self)}')
            self.get_file().close()
            global _grib_files_closed
            _grib_files_closed += 1

    def __iter__(self):
        return self

    def __next__(self):
        # load a next message from self.file
        message_id = ecc.codes_grib_new_from_file(self.get_file())
        if message_id is not None:
            return GribMessageWithCache(message_id, self)
        else:
            raise StopIteration

    def __str__(self):
        return self._filename

    def __repr__(self):
        dump = [repr(type(self))]
        dump.append(f'GRIB file {self._filename}')
        return '\n'.join(dump)


class GribFileIndexedBy(abstract_dictionary.AbstractDictionary, GribAbstractItem):
    def __init__(self, filename, *keys):
        super().__init__()
        if len(keys) == 0:
            raise Exception('index must contain at least one key')
        self._keys = keys
        self._filename = filename
        self._id = ecc.codes_index_new_from_file(self._filename, self._keys)
        global _grib_indices
        _grib_indices += 1

    def get_id(self):
        if self._id is None:
            raise ValueError('GRIB index already released')
        return self._id

    def __del__(self):
        if self._id is not None:
            logger.info(f'releasing GribFileIndexedBy id={self.get_id()}, filename={self._filename}')
            ecc.codes_index_release(self.get_id())
            self._id = None
            global _grib_indices_released
            _grib_indices_released += 1

    def __getitem__(self, index):
        index = utils.to_tuple(index)
        if len(index) != len(self._keys):
            raise KeyError(f'expected number of indices is {len(self._keys)}')
        for key, index in zip(self._keys, index):
            ecc.codes_index_select(self.get_id(), key, index)
        value = list(_GribMessagesFromIndexGenerator(self))
        if len(value) == 0:
            raise KeyError(f'no GRIB messages for {self._keys}={index}')
        return value

    def __str__(self):
        return f'{self._filename}, keys: {self._keys}'

    def __repr__(self):
        dump = [repr(type(self))]
        dump.append(f'GRIB file {self._filename}')
        dump.append(f'\tIndices: {self._keys}')
        return '\n'.join(dump)

    def get_indices(self, key):
        if key in self._keys:
            return ecc.codes_index_get(self.get_id(), key)
        else:
            raise KeyError(f'Invalid key={key}. Available keys={self._keys}')


class GribFileIndexedByWithCache(abstract_dictionary.AbstractCacheDictionary, GribFileIndexedBy):
    pass


class GribFileUniquelyIndexedBy(GribFileIndexedBy):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        if len(value) > 1:
            raise KeyError(f'There is more than one GRIB message for {self._keys}={index}')
        return value[0]


class GribFileUniquelyIndexedByWithCache(abstract_dictionary.AbstractCacheDictionary, GribFileUniquelyIndexedBy):
    pass


class _GribMessagesFromIndexGenerator:
    def __init__(self, grib_file_indexed):
        self._grib_file_indexed = grib_file_indexed

    def __iter__(self):
        return self

    def __next__(self):
        message_id = ecc.codes_new_from_index(self._grib_file_indexed.get_id())
        if message_id is not None:
            return GribMessageWithCache(message_id, self._grib_file_indexed)
        else:
            raise StopIteration
