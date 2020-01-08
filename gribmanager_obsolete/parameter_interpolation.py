import math
import sys

_eps = sys.float_info.epsilon


class Longitude:
    def __init__(self, lon):
        self._lon = lon

    def dist(self, lon):
        delta = abs(lon - self._lon) % 360.
        return delta if delta <= 180. else 360. - delta


def linear_interpolation(x0, xys):
    (x1, y1), (x2, y2) = xys
    if isinstance(x1, Longitude):
        dist_x0_x1 = x1.dist(x0)
        dist_x0_x2 = x2.dist(x0)
    else:
        dist_x0_x1 = abs(x1 - x0)
        dist_x0_x2 = abs(x2 - x0)
    dist = dist_x0_x1 + dist_x0_x2
    #FIXME: there is a problem when dist is a pandas.Timedelta object
    p = dist_x0_x1 / dist #if dist > _eps else .5
    return (1-p) * y1 + p * y2


class InterpolationNode:
    def __init__(self, label, value=math.nan, parent=None):
        self._label = label
        self._value = value
        self._children = []
        self._parent = parent
        if self._parent is not None:
            self._parent._add_child(self)

    def _add_child(self, child):
        self._children.append(child)

    def interpolate(self, labels):
        if len(labels) > 0:
            label, *remaining_labels = labels
            labels_values = [(child._label, child.interpolate(remaining_labels)) for child in self._children]
            if len(labels_values) == 2:
                return linear_interpolation(label, labels_values)
            elif len(labels_values) == 1:
                _, value = labels_values[0]
                return value
            else:
                raise ValueError(f'cannot interpolate from {len(labels_values)} values')
        else:
            return self._value
