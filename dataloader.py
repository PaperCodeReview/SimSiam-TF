import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from augment import Augment


AUTO = tf.data.experimental.AUTOTUNE


def set_dataset(task, data_path):
    trainset = pd.read_csv(
        os.path.join(
            data_path, 'imagenet_trainset.csv'
        )).values.tolist()
    trainset = [[os.path.join(data_path, t[0]), t[1]] for t in trainset]

    if task == 'lincls':
        valset = pd.read_csv(
            os.path.join(
                data_path, 'imagenet_valset.csv'
            )).values.tolist()
        valset = [[os.path.join(data_path, t[0]), t[1]] for t in valset]
        return np.array(trainset, dtype='object'), np.array(valset, dtype='object')

    return np.array(trainset, dtype='object')


class DataLoader:
    def __init__(self, args, mode, datalist, batch_size, num_workers=1, shuffle=True):
        self.args = args
        self.mode = mode
        self.datalist = datalist
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.augset = Augment(self.args, self.mode)
        self.dataloader = self._dataloader()

    def __len__(self):
        return len(self.datalist)

    def fetch_dataset(self, path, y=None):
        x = tf.io.read_file(path)
        if y is not None:
            return tf.data.Dataset.from_tensors((x, y))
        return tf.data.Dataset.from_tensors(x)

    def augmentation(self, img, shape):
        if self.args.task == 'pretext':
            img_list = []
            for _ in range(2): # query, key
                aug_img = tf.identity(img)
                aug_img = self.augset._augment_simsiam(aug_img, shape)
                img_list.append(aug_img)
            return img_list
        else:
            return self.augset._augment_lincls(img, shape)

    def dataset_parser(self, value, label=None):
        shape = tf.image.extract_jpeg_shape(value)
        img = tf.io.decode_jpeg(value, channels=3)
        if label is None:
            # moco
            return self.augmentation(img, shape)
        else:
            # lincls
            inputs = self.augmentation(img, shape)
            labels = tf.one_hot(label, self.args.classes)
            return (inputs, labels)
        
    def _dataloader(self):
        self.imglist = self.datalist[:,0].tolist()
        if self.args.task == 'pretext':
            dataset = tf.data.Dataset.from_tensor_slices(self.imglist)
        else:
            self.labellist = self.datalist[:,1].tolist()
            dataset = tf.data.Dataset.from_tensor_slices((self.imglist, self.labellist))

        dataset = dataset.repeat()
        if self.shuffle:
            dataset = dataset.shuffle(len(self.datalist))

        dataset = dataset.interleave(self.fetch_dataset, num_parallel_calls=AUTO)
        dataset = dataset.map(self.dataset_parser, num_parallel_calls=AUTO)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTO)
        return dataset