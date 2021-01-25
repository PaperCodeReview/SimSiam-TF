import tensorflow as tf
from tensorflow.keras.initializers import Constant


class Conv2D(tf.keras.layers.Conv2D):
    def build(self, input_shape):
        k = 1 / input_shape[-1]
        self.kernel_initializer = Constant(tf.random.uniform([], -tf.sqrt(k), tf.sqrt(k)))
        super(Conv2D, self).build(input_shape)


class Dense(tf.keras.layers.Dense):
    def build(self, input_shape):
        k = 1 / input_shape[-1]
        self.kernel_initializer = Constant(tf.random.uniform([], -tf.sqrt(k), tf.sqrt(k)))
        super(Dense, self).build(input_shape)