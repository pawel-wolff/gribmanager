import logging
import itertools
import pandas as pd


_ul = logging.getLogger(__name__)
_ul.addHandler(logging.NullHandler())
_logger_configured = False


def _logger_config(logging_level):
    global _logger_configured
    if not _logger_configured:
        _ul.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - in %(module)s.%(funcName)s (line %(lineno)d): %(message)s')
        console_handler.setFormatter(formatter)
        _ul.addHandler(console_handler)
        _logger_configured = True


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


def unique(an_iterable):
    iterator = iter(an_iterable)
    try:
        item = next(iterator)
    except StopIteration:
        raise ValueError('an iterable has no elements')
    try:
        next(iterator)
        raise ValueError('an iterator has more than one element')
    except StopIteration:
        return item


def to_tuple(indices):
    if not isinstance(indices, tuple):
        return indices,
    return indices


def linear_search(x, an_iterable):
    """

    :param x:
    :param an_iterable:
    :return: a smallest i for which the i-th element of an_iterable is > x
    """
    for i, v in zip(itertools.count(), an_iterable):
        if v > x:
            return i
    raise ValueError(f'no element of {an_iterable} is > than {x}')


def sandwiching_values(x, an_iterable):
    """

    :param x:
    :param an_iterable: must be a non-decreasing sequence
    :return: a smallest i for which the i-th element of an_iterable is > x
    """
    it = iter(zip(itertools.count(), an_iterable))
    try:
        i, v = next(it)
        if v > x:
            raise ValueError(f'all elements of an_iterable are > x={x}')
        v_new = None
        for i, v_new in it:
            if x <= v_new:
                return (i-1, v), (i, v_new)
            v = v_new
        if v_new is None:
            raise ValueError(f'an_iterable has only one element')
        raise ValueError(f'all elements of an_iterable are < x={x}')
    except StopIteration:
        raise ValueError(f'an_iterable has no element')


def sandwiching_values_by_binary_search(x, a, b, f, aux=None):
    if a > b:
        raise ValueError(f'there are no elements; (lat,lon)={aux}')
    if x < f(a):
        _ul.warning(f'smallest element={f(a)} is > x={x}; (lat,lon)={aux}')
        return (a, f(a)),
    if f(b) < x:
        _ul.warning(f'greatest element={f(b)} is < x={x}; (lat,lon)={aux}')
        return (b, f(b)),
    if a == b:
        return (a, f(a)),
    else:
        return _rec_sandwiching_values_by_binary_search(x, a, b, f)


def strict_sandwiching_values_by_binary_search(x, a, b, f):
    if a >= b:
        raise ValueError(f'there is at most one element')
    if x < f(a):
        raise ValueError(f'all elements are > x={x}')
    if f(b) < x:
        raise ValueError(f'all elements are < x={x}')
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
        return (a, f(a)), (b, f(b))
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
        return __repr__(self)

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


def get_timestamp_for_ENfilename(filename):
    year = int(filename[-8:-6])
    if year > 70:
        year += 1900
    else:
        year += 2000
    month = int(filename[-6:-4])
    day = int(filename[-4:-2])
    hour = int(filename[-2:])
    return pd.Timestamp(year, month, day, hour)
