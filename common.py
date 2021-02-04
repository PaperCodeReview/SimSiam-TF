import os
import sys
import yaml
import random
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime


def check_arguments(args):
    assert args.src_path is not None, 'src_path must be entered.'
    assert args.data_path is not None, 'data_path must be entered.'
    assert args.result_path is not None, 'result_path must be entered.'
    return args


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",           type=str,       default='pretext',
                        choices=['pretext', 'lincls'])
    parser.add_argument("--dataset",        type=str,       default='imagenet')
    parser.add_argument("--freeze",         action='store_true')
    parser.add_argument("--backbone",       type=str,       default='resnet50')
    parser.add_argument("--batch_size",     type=int,       default=256)
    parser.add_argument("--classes",        type=int,       default=1000)
    parser.add_argument("--img_size",       type=int,       default=224)
    parser.add_argument("--proj_dim",       type=int,       default=2048)
    parser.add_argument("--proj_bn_hidden", action='store_true')
    parser.add_argument("--proj_bn_output", action='store_true')
    parser.add_argument("--pred_dim",       type=int,       default=512)
    parser.add_argument("--pred_bn_hidden", action='store_true')
    parser.add_argument("--pred_bn_output", action='store_true')

    parser.add_argument("--weight_decay",   type=float,     default=0.)
    parser.add_argument("--use_bias",       action='store_true')
    parser.add_argument("--steps",          type=int,       default=0)
    parser.add_argument("--epochs",         type=int,       default=200)

    parser.add_argument("--evaluate",       action='store_true')
    parser.add_argument("--checkpoint",     action='store_true')
    parser.add_argument("--history",        action='store_true')
    parser.add_argument("--tensorboard",    action='store_true')
    parser.add_argument("--tb_interval",    type=int,       default=0)
    parser.add_argument("--tb_histogram",   type=int,       default=0)
    parser.add_argument("--lr_mode",        type=str,       default='cosine',  
                        choices=['constant', 'cosine'])

    parser.add_argument('--src_path',       type=str,       default='.')
    parser.add_argument('--data_path',      type=str,       default=None)
    parser.add_argument('--result_path',    type=str,       default='./result')
    parser.add_argument('--snapshot',       type=str,       default=None)
    parser.add_argument("--gpus",           type=str,       default='-1')
    parser.add_argument("--summary",        action='store_true')
    parser.add_argument("--resume",         action='store_true')
    parser.add_argument("--ignore-search",  type=str,       default='')

    return check_arguments(parser.parse_args())


def set_seed(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)
    return logger


def get_session(args):
    assert int(tf.__version__.split('.')[0]) >= 2.0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.gpus != '-1':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


def create_stamp():
    weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    temp = datetime.now()
    return "{:02d}{:02d}{:02d}_{}_{:02d}_{:02d}_{:02d}".format(
        temp.year % 100,
        temp.month,
        temp.day,
        weekday[temp.weekday()],
        temp.hour,
        temp.minute,
        temp.second,
    )


def search_same(args):
    search_ignore = ['evaluate', 'checkpoint', 'history', 'tensorboard', 
                     'tb_interval', 'snapshot', 'summary',
                     'src_path', 'data_path', 'result_path', 
                     'resume', 'stamp', 'gpus', 'ignore_search']
    if len(args.ignore_search) > 0:
        search_ignore += args.ignore_search.split(',')

    initial_epoch = 0
    stamps = os.listdir(f'{args.result_path}/{args.task}')
    for stamp in stamps:
        try:
            desc = yaml.full_load(
                open(f'{args.result_path}/{args.task}/{stamp}/model_desc.yml', 'r'))
        except:
            continue

        flag = True
        for k, v in vars(args).items():
            if k in search_ignore:
                continue

            if k == 'dataset' and k not in desc:
                desc[k] = 'imagenet'
                
            if v != desc[k]:
                # if stamp == '210120_Wed_05_19_52':
                #     print(stamp, k, desc[k], v)
                flag = False
                break
        
        if flag:
            args.stamp = stamp
            df = pd.read_csv(
                os.path.join(
                    args.result_path, 
                    f'{args.task}/{args.stamp}/history/epoch.csv'))

            if len(df) > 0:
                if int(df['epoch'].values[-1]+1) == args.epochs:
                    print(f'{stamp} Training already finished!!!')
                    return args, -1

                elif np.isnan(df['loss'].values[-1]) or np.isinf(df['loss'].values[-1]):
                    print('{} | Epoch {:04d}: Invalid loss, terminating training'.format(stamp, int(df['epoch'].values[-1]+1)))
                    return args, -1

                else:
                    ckpt_list = sorted(
                        [d.split('.index')[0] for d in os.listdir(
                            f'{args.result_path}/{args.task}/{args.stamp}/checkpoint') if 'index' in d])
                    
                    if len(ckpt_list) > 0:
                        args.snapshot = f'{args.result_path}/{args.task}/{args.stamp}/checkpoint/{ckpt_list[-1]}'
                        initial_epoch = int(df['epoch'].iloc[-1]) + 1
                    else:
                        print('{} Training already finished!!!'.format(stamp))
                        return args, -1
            break
    return args, initial_epoch