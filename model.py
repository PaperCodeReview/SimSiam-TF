import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense


WEIGHTS_HASHES = {'resnet50' : '4d473c1dd8becc155b73f8504c6f6626',}
MODEL_DICT = {'resnet50' : tf.keras.applications.ResNet50,}
FAMILY_DICT = {'resnet50' : tf.python.keras.applications.resnet,}
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


def set_lincls(args, backbone):
    DEFAULT_ARGS = {
        "use_bias": args.use_bias,
        "kernel_regularizer": l2(args.weight_decay)}
    
    if args.freeze:
        backbone.trainable = False
        
    x = backbone.get_layer(name='avg_pool').output
    x = _dense(**DEFAULT_ARGS)(args.classes, name='predictions')(x)
    model = Model(backbone.input, x, name='lincls')
    return model


class SimSiam(Model):
    def __init__(self, args, logger, num_workers=1, **kwargs):
        super(SimSiam, self).__init__(**kwargs)
        self.args = args
        self._num_workers = num_workers
        norm = 'bn' if self._num_workers == 1 else 'syncbn'
        
        DEFAULT_ARGS = {
            "use_bias": self.args.use_bias,
            "kernel_regularizer": l2(self.args.weight_decay)}
        FAMILY_DICT[self.args.backbone].Conv2D = _conv2d(**DEFAULT_ARGS)
        FAMILY_DICT[self.args.backbone].BatchNormalization = _batchnorm(norm=norm)
        FAMILY_DICT[self.args.backbone].Dense = _dense(**DEFAULT_ARGS)

        backbone = MODEL_DICT[self.args.backbone](
            include_top=False,
            weights=None,
            input_shape=(self.args.img_size, self.args.img_size, 3),
            pooling='avg')
        
        x = backbone.output

        # Projection MLP
        for i in range(2):
            x = _dense(**DEFAULT_ARGS)(self.args.proj_dim, name=f'proj_fc{i+1}')(x)
            if self.args.proj_bn_hidden:
                x = _batchnorm(norm=norm)(epsilon=1.001e-5, name=f'proj_bn{i+1}')(x)
            x = Activation('relu', name=f'proj_relu{i+1}')(x)

        x_proj = _dense(**DEFAULT_ARGS)(self.args.proj_dim, name='proj_fc3')(x)
        if self.args.proj_bn_output:
            x = _batchnorm(norm=norm)(epsilon=1.001e-5, name='proj_bn3')(x)
        
        self.encoder = Model(backbone.input, [x_proj, x], name='encoder')

        # Prediction MLP
        self.prediction_MLP = Sequential()
        self.prediction_MLP.add(_dense(**DEFAULT_ARGS)(self.args.pred_dim, name='pred_fc1'))
        if self.args.pred_bn_hidden:
            self.prediction_MLP.add(_batchnorm(norm=norm)(epsilon=1.001e-5, name='pred_bn1'))
        self.prediction_MLP.add(Activation('relu', name='pred_relu1'))

        self.prediction_MLP.add(_dense(**DEFAULT_ARGS)(self.args.proj_dim, name='pred_fc2'))
        if self.args.pred_bn_output:
            self.prediction_MLP.add(_batchnorm(norm=norm)(epsilon=1.001e-5, name='pred_bn2'))
        
        # Load checkpoints
        if self.args.snapshot:
            self.load_weights(self.args.snapshot)
            logger.info('Load weights at {}'.format(self.args.snapshot))

    def compile(
        self,
        optimizer,
        loss,
        run_eagerly=None):

        super(SimSiam, self).compile(
            optimizer=optimizer, run_eagerly=run_eagerly)

        self._loss = loss
        self._replica_context = tf.distribute.get_replica_context()

    def train_step(self, data):
        imgs = tf.concat(data, axis=0)
        with tf.GradientTape() as tape:
            z = tf.cast(self.encoder(imgs, training=True), tf.float32)
            z1, z2 = tf.split(z, num_or_size_splits=2, axis=0)

            p = tf.cast(self.prediction_MLP(z, training=True), tf.float32)
            p1, p2 = tf.split(p, num_or_size_splits=2, axis=0)

            loss_simsiam = (self._loss(p1, tf.stop_gradient(z2)) + self._loss(p2, tf.stop_gradient(z1))) / 2
            loss_simsiam = tf.reduce_mean(loss_simsiam)
            loss_decay = sum(self.encoder.losses + self.prediction_MLP.losses)

            loss = loss_simsiam + loss_decay
            total_loss = loss / self._num_workers

        trainable_vars = self.encoder.trainable_variables + self.prediction_MLP.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        results = {'loss': loss, 'loss_simsiam': loss_simsiam, 'weight_decay': loss_decay}
        return results