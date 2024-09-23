import sys
import math
import pandas as pd


_eps = 10. * sys.float_info.epsilon


def ensure_tuple(indices):
    if not isinstance(indices, tuple):
        return indices,
    return indices


def groupby(an_iterable, key_func):
    """
    Group an iterable according to a result on key_func

    :param an_iterable: an iterable to group
    :param key_func: a callable
    :return: a dictionary (key, list of items)
    """

    group_by_key = {}
    for item in an_iterable:
        k = key_func(item)
        group_by_key.setdefault(k, []).append(item)
    return group_by_key


def normalize_longitude(arr, smallest_lon_coord=-180.):
    return (arr - smallest_lon_coord) % 360. + smallest_lon_coord


class Longitude(float):
    def __new__(cls, lon):
        return float.__new__(cls, normalize_longitude(float(lon)))

    def __add__(self, other):
        return Longitude(float.__add__(self, other))

    def __sub__(self, other):
        return normalize_longitude(float.__sub__(self, other))


def midpoint(u, v, p=0.5):
    """
    :return: (1-p)*u + p*v
    """
    return u + p * (v - u)


def linear_interpolation(x0, xys):
    (x1, y1), (x2, y2) = xys
    if isinstance(x0, pd.Timestamp):
        x0, x1, x2 = x0.timestamp(), x1.timestamp(), x2.timestamp()
    dist_x0_x1 = abs(x1 - x0)
    dist_x0_x2 = abs(x2 - x0)
    dist = dist_x0_x1 + dist_x0_x2
    assert math.isclose(dist, abs(x2 - x1)), (x0, x1, x2)
    p = dist_x0_x1 / dist if dist > _eps else .5
    return midpoint(y1, y2, p)



class AbstractDictionary:
    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def __getitem__(self, key):
        raise NotImplementedError

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class AbstractCacheDictionary(AbstractDictionary):
    def __init__(self, *args, **kwargs):
        self._dict = {}
        self._keys = set()
        self._keys_not_found = set()
        super().__init__(*args, **kwargs)

    def __contains__(self, key):
        if key in self._dict or key in self._keys:
            return True
        elif key in self._keys_not_found:
            return False
        else:
            contains = super().__contains__(key)
            if contains:
                self._keys.add(key)
            else:
                self._keys_not_found.add(key)
            return contains

    def __getitem__(self, key):
        if key not in self._dict:
            value = super().__getitem__(key)
            self._dict[key] = value
            return value
        else:
            return self._dict[key]

    def get_cache(self):
        return list(self._dict.values())


def sandwiching_values_by_binary_search(x, a, b, f, aux=None):
    if a > b:
        raise ValueError(f'there are no elements; (lat,lon)={aux}')
    if x < f(a):
        # TODO: uncomment the line below
        #logger.warning(f'smallest element={f(a)} is > x={x}; (lat,lon)={aux}')
        return (a, ), (f(a), )
    if f(b) < x:
        # TODO: uncomment the line below
        #logger.warning(f'greatest element={f(b)} is < x={x}; (lat,lon)={aux}')
        return (b, ), (f(b), )
    if a == b:
        return (a, ), (f(a), )
    else:
        return _rec_sandwiching_values_by_binary_search(x, a, b, f)


def _rec_sandwiching_values_by_binary_search(x, a, b, f):
    """
    Assumes that a+1 <= b

    :param x:
    :param a:
    :param b:
    :param f:
    :return:
    """
    if a + 1 == b:
        return (a, b), (f(a), f(b))
    c = (a + b) // 2
    if x <= f(c):
        return _rec_sandwiching_values_by_binary_search(x, a, c, f)
    else:
        return _rec_sandwiching_values_by_binary_search(x, c, b, f)


class NearestPoint:
    def __init__(self, lat, lon, index):
        self.lat = lat
        self.lon = lon
        self.index = index

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'[lat: {self.lat}, lon: {self.lon}, index: {self.index}]'


def four_nearest_points_in_rectangular_grid(x0, y0, x1, y1, dx, dy, nx, ny, x_is_major, x, y):
    if not (dx > 0 and x0 <= x <= x1) and not (dx < 0 and x0 >= x >= x1):
        raise ValueError(f'latitude {x} out of range [{x0}, {x1}]')
    # TODO: similar check for longitudes, but be careful
    x_upper = max(x0, x1)
    x_lower = min(x0, x1)
    x = min(x, x_upper - abs(dx) / 2.)
    x = max(x, x_lower + abs(dx) / 2.)
    i = int((x - x0) // dx)
    j = int(((y - y0) % (360. if dy > 0 else -360.)) // dy)
    grid_lat = x0 + i * dx
    grid_lon = (y0 + j * dy) % 360.
    if grid_lon > 180.:
        grid_lon -= 360.
    if x_is_major:
        major_offset = ny * i
        return (NearestPoint(grid_lat, grid_lon, major_offset + j),
                NearestPoint(grid_lat, grid_lon + dy, major_offset + (j + 1) % ny)), \
               (NearestPoint(grid_lat + dx, grid_lon, major_offset + ny + j),
                NearestPoint(grid_lat + dx, grid_lon + dy, major_offset + ny + (j + 1) % ny))
    else:
        major_offset = nx * j
        major_offset_plus_1 = nx * ((j + 1) % ny)
        return (NearestPoint(grid_lat, grid_lon, major_offset + i),
                NearestPoint(grid_lat, grid_lon + dy, major_offset_plus_1 + i)), \
               (NearestPoint(grid_lat + dx, grid_lon, major_offset + i + 1),
                NearestPoint(grid_lat + dx, grid_lon + dy, major_offset_plus_1 + i + 1))
