import logging
import itertools

import gribmanager as gm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)


INPUT = '/home/wolp/data/ECMWF/EN19090900'
#INPUT_1994 = '/home/wolp/data/MCMWF-ENFILES/EN94070100'
#INPUT = '/home/wolp/data/MCMWF-ENFILES/EN16070200.bis'


if __name__ == '__main__':
    idx = gm.open_grib(INPUT, [gm.PARAMETER_ID, gm.TYPE_OF_LEVEL])
    params = idx.get_indices(gm.PARAMETER_ID)
    type_of_levels = idx.get_indices(gm.TYPE_OF_LEVEL)

    for p, l in itertools.product(params, type_of_levels):
        try:
            msg = idx[p, l]
        except KeyError:
            continue
        print(f'{p}, {l}: {len(msg)}')
