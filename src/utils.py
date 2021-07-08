import os, sys, shutil
from datetime import datetime
import logging
from collections import namedtuple
import pandas as pd

def get_logger(model_name):
    logger = logging.getLogger('{:%y%m%d_%H%M%S}'.format(datetime.now()))
    logger.setLevel(logging.DEBUG)
    handler_format = logging.Formatter('%(asctime)s - %(message)s', "%y/%m/%d %H:%M:%S")

    if model_name == 'test':
        dir_path = os.path.join('log', 'test')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(handler_format)
        logger.addHandler(console_handler)
    else:
        now = '{:%y%m%d_%H%M%S}'.format(datetime.today())
        dir_path = os.path.join('log', f'{now}_{model_name}')

    if os.path.exists(dir_path): shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    shutil.copytree('src', os.path.join(dir_path, 'src'))
    shutil.copy('train.py', dir_path)

    debug_file_handler = logging.FileHandler(os.path.join(dir_path, 'debug.log'))
    debug_file_handler.setLevel(logging.INFO)
    debug_file_handler.setFormatter(handler_format)
    logger.addHandler(debug_file_handler)

    return logger, dir_path

def iterate(d, param={}):
    d, param = d.copy(), param.copy()
    d_list = []

    for k, v in d.items():
        if isinstance(v, list):
            for vi in v:
                d[k], param[k] = vi, vi
                d_list += iterate(d, param)
            return d_list

        if isinstance(v, dict):
            add_d_list = iterate(v, param)
            if len(add_d_list) > 1:
                for vi, pi in add_d_list:
                    d[k] = vi
                    d_list += iterate(d, pi)
                return d_list

    return [[d, param]]

def dict_to_tuple(dict_data):
    return namedtuple('X', dict_data.keys())(*tuple(map(lambda x: x if not isinstance(x, dict) else dict_to_tuple(x), dict_data.values())))

def record_score(d, path):
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df = df.append(d, ignore_index=True)
    else:
        df = pd.DataFrame(d.values(), index=d.keys()).T
    df.to_csv(path)