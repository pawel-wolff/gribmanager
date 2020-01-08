import gribmanager_obsolete as gm
import random
import cProfile
import pstats
from pstats import SortKey
import os
import psutil


INPUT = '/home/wolp/data/ECMWF/EN19090900'
INPUT_1994 = '/home/wolp/data/ECMWF/EN94070100'


def profile_four_nearest_points(param_id, n, ecc=False):
    for i in range(n):
        level = random.randint(1, 137)
        m = g[param_id, level]
        p = m.get_four_nearest_points(random.uniform(-90., 90.), random.uniform(-180., 180.), ecc)


process = psutil.Process(os.getpid())
print(process.memory_info().rss / 1024 / 1024)  # in MB

for i in range(5):
    print(i)
    g = gm.open_grib(INPUT, index_keys=['paramId', 'level'], unique_indexing=True)
    profile_four_nearest_points(131, 10000)
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 1024 / 1024)  # in MB

cProfile.run('profile_four_nearest_points(131, 10000)', 'my_stats')
p = pstats.Stats('my_stats')
p.strip_dirs().sort_stats(SortKey.TIME).print_stats()
