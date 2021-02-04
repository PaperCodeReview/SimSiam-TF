import os
import six
import yaml
import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.experimental import CosineDecay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from common import create_stamp


class OptionalLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, args, steps_per_epoch, initial_epoch):
        super(OptionalLearningRateSchedule, self).__init__()
        self.args = args
        self.steps_per_epoch = steps_per_epoch
        self.initial_epoch = initial_epoch

        if self.args.lr_mode == 'cosine':
            self.lr_scheduler = CosineDecay(self.args.lr, self.args.epochs)

        elif self.args.lr_mode == 'constant':
            self.lr_scheduler = lambda x: self.args.lr
            

    def get_config(self):
        return {
            'steps_per_epoch': self.steps_per_epoch,
            'init_lr': self.args.lr,
            'lr_mode': self.args.lr_mode,
            'lr_value': self.args.lr_value,
            'lr_interval': self.args.lr_interval,}

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        step += self.initial_epoch * self.steps_per_epoch
        lr_epoch = (step / self.steps_per_epoch)
        if self.args.lr_mode == 'constant':
            return self.args.lr
        else:
            return self.lr_scheduler(lr_epoch)


class CustomCSVLogger(CSVLogger):
    """Save averaged logs during training.
    """
    def on_epoch_begin(self, epoch, logs=None):
        self.batch_logs = {}

    def on_batch_end(self, batch, logs=None):
        for k, v in logs.items():
            if k not in self.batch_logs:
                self.batch_logs[k] = [v]
            else:
                self.batch_logs[k].append(v)

    def on_epoch_end(self, epoch, logs=None):
        final_logs = {k: np.mean(v) for k, v in self.batch_logs.items()}
        super(CustomCSVLogger, self).on_epoch_end(epoch, final_logs)


def create_callbacks(args, logger, initial_epoch):
    if not args.resume:
        if args.checkpoint or args.history or args.tensorboard:
            if os.path.isdir(f'{args.result_path}/{args.task}/{args.stamp}'):
                flag = input(f'\n{args.task}/{args.stamp} is already saved. '
                              'Do you want new stamp? (y/n) ')
                if flag == 'y':
                    args.stamp = create_stamp()
                    initial_epoch = 0
                    logger.info(f'New stamp {args.stamp} will be created.')
                elif flag == 'n':
                    return -1, initial_epoch
                else:
                    logger.info(f'You must select \'y\' or \'n\'.')
                    return -2, initial_epoch

            os.makedirs(f'{args.result_path}/{args.task}/{args.stamp}')
            yaml.dump(
                vars(args), 
                open(f'{args.result_path}/{args.task}/{args.stamp}/model_desc.yml', 'w'), 
                default_flow_style=False)
        else:
            logger.info(f'{args.stamp} is not created due to '
                        f'checkpoint - {args.checkpoint} | '
                        f'history - {args.history} | '
                        f'tensorboard - {args.tensorboard}')

    callbacks = []
    if args.checkpoint:
        if args.task == 'pretext':
            callbacks.append(ModelCheckpoint(
                filepath=f'{args.result_path}/{args.task}/{args.stamp}/checkpoint/latest',
                monitor='loss',
                mode='min',
                verbose=1,
                save_weights_only=True))
            callbacks.append(ModelCheckpoint(
                filepath=f'{args.result_path}/{args.task}/{args.stamp}/checkpoint/best',
                monitor='loss',
                mode='min',
                verbose=1,
                save_weights_only=True,
                save_best_only=True))
        else:
            callbacks.append(ModelCheckpoint(
                filepath=f'{args.result_path}/{args.task}/{args.stamp}/checkpoint/latest',
                monitor='val_acc1',
                mode='max',
                verbose=1,
                save_weights_only=True))
            callbacks.append(ModelCheckpoint(
                filepath=f'{args.result_path}/{args.task}/{args.stamp}/checkpoint/best',
                monitor='val_acc1',
                mode='max',
                verbose=1,
                save_weights_only=True,
                save_best_only=True))

    if args.history:
        os.makedirs(f'{args.result_path}/{args.task}/{args.stamp}/history', exist_ok=True)
        callbacks.append(CustomCSVLogger(
            filename=f'{args.result_path}/{args.task}/{args.stamp}/history/epoch.csv',
            separator=',', append=True))

    if args.tensorboard:
        callbacks.append(TensorBoard(
            log_dir=f'{args.result_path}/{args.task}/{args.stamp}/logs',
            histogram_freq=args.tb_histogram,
            write_graph=True, 
            write_images=True,
            update_freq=args.tb_interval,
            profile_batch=100,))

    return callbacks, initial_epoch