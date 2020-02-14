import time
from common.log import logger
from common import log
import logging
import gribmanager as gm

log.start_logging('/home/wolp/data/tmp/log.txt', logging.INFO)
idx = gm.open_grib('/home/wolp/data/ECMWF/EN19061918', index_keys=[gm.PARAMETER_ID])
p129 = idx[129]
logger.info(f'p129 with len={len(p129)} allocated')
idx = None
logger.info('idx disposed')
v = p129[0].get_value_at(30., 40.)
logger.info(f'v={v}')
p129 = None
logger.info('p129 disposed')
logger.info('sleep begin')
time.sleep(10)
logger.info('sleep end')
