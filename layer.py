import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.initializers import Constant


BatchNorm_DICT = {
    "bn": BatchNormalization,
    "syncbn": SyncBatchNormalization}


def _conv2d(**custom_kwargs):
    def _func(*args, **kwargs):
        kwargs.update(**custom_kwargs)
        return Conv2D(*args, **kwargs)
    return _func


def _batchnorm(norm='bn', **custom_kwargs):
    def _func(*args, **kwargs):
        kwargs.update(**custom_kwargs)
        return BatchNorm_DICT[norm](*args, **kwargs)
    return _func


def _dense(**custom_kwargs):
    def _func(*args, **kwargs):
        kwargs.update(**custom_kwargs)
        return Dense(*args, **kwargs)
    return _func


# class Conv2D(tf.keras.layers.Conv2D):
#     def build(self, input_shape):
#         k = 1 / input_shape[-1]
#         self.kernel_initializer = Constant(tf.random.uniform([], -tf.sqrt(k), tf.sqrt(k)))
#         super(Conv2D, self).build(input_shape)


# class Dense(tf.keras.layers.Dense):
#     def build(self, input_shape):
#         k = 1 / input_shape[-1]
#         self.kernel_initializer = Constant(tf.random.uniform([], -tf.sqrt(k), tf.sqrt(k)))
#         super(Dense, self).build(input_shape)