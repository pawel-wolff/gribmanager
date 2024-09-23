import logging
import random
import cProfile
import pstats
from pstats import SortKey
import os
import psutil

from gribmanager import grib_manager as gm, parameter_manager as pm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test(p_m, param_id, n):
    for i in range(n):
        p = random.uniform(100, 50000)
        lat = random.uniform(-90, 90)
        lon = random.uniform(-180, 180)
        p_m.get_parameter(param_id).get_value_at(lat, lon, p)


INPUTs = ['/home/wolp/data/ECMWF/EN19090900', '/home/wolp/data/ECMWF/EN19090903']


if __name__ == '__main__':
    p_m = pm.ParameterManager(INPUTs[0])

    process = psutil.Process(os.getpid())
    logger.info(process.memory_info().rss / 1024 / 1024)  # in MB

    cProfile.run('test(p_m, 132, 10000)', 'my_stats')
    p = pstats.Stats('my_stats')
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats()

    process = psutil.Process(os.getpid())
    logger.info(process.memory_info().rss / 1024 / 1024)  # in MB
