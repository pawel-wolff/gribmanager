import time
import logging
import gribmanager as gm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
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
