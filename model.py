import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Add
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant

from resnet import ResNet18
from layer import _conv2d
from layer import _batchnorm
from layer import _dense


MODEL_DICT = {
    'resnet18' : ResNet18,
    'resnet50' : tf.keras.applications.ResNet50,}
FAMILY_DICT = {
    'resnet18' : tf.python.keras.applications.resnet,
    'resnet50' : tf.python.keras.applications.resnet,}


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

        DEFAULT_ARGS.update({'norm': norm})
        backbone = MODEL_DICT[self.args.backbone](
            include_top=False,
            weights=None,
            input_shape=(self.args.img_size, self.args.img_size, 3),
            pooling='avg',
            **DEFAULT_ARGS if self.args.backbone == 'resnet18' else {})
        DEFAULT_ARGS.pop('norm')

        x = backbone.output
        outputs = []

        # Projection MLP
        num_mlp = 3 if self.args.dataset == 'imagenet' else 2
        for i in range(num_mlp-1):
            x = _dense(**DEFAULT_ARGS)(self.args.proj_dim, name=f'proj_fc{i+1}')(x)
            if self.args.proj_bn_hidden:
                x = _batchnorm(norm=norm)(epsilon=1.001e-5, name=f'proj_bn{i+1}')(x)
            x = Activation('relu', name=f'proj_relu{i+1}')(x)

        x = _dense(**DEFAULT_ARGS)(self.args.proj_dim, name='proj_fc3')(x)
        if self.args.proj_bn_output:
            x = _batchnorm(norm=norm)(epsilon=1.001e-5, name='proj_bn3')(x)
        
        outputs.append(x)

        # Prediction MLP
        x = _dense(**DEFAULT_ARGS)(self.args.pred_dim, name='pred_fc1')(x)
        if self.args.pred_bn_hidden:
            x = _batchnorm(norm=norm)(epsilon=1.001e-5, name='pred_bn1')(x)
        x = Activation('relu', name='pred_relu1')(x)

        x = _dense(**DEFAULT_ARGS)(self.args.proj_dim, name='pred_fc2')(x)
        if self.args.pred_bn_output:
            x = _batchnorm(norm=norm)(epsilon=1.001e-5, name='pred_bn2')(x)

        outputs.append(x)
        self.encoder = Model(backbone.input, outputs, name='encoder')
        
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

    def train_step(self, data):
        img1, img2 = data
        with tf.GradientTape() as tape:
            z1, p1 = self.encoder(img1, training=True)
            z2, p2 = self.encoder(img2, training=True)
            
            if self.args.stop_gradient:
                loss_simsiam = (self._loss(p1, tf.stop_gradient(z2)) + self._loss(p2, tf.stop_gradient(z1))) / 2
            else:
                loss_simsiam = (self._loss(p1, z2) + self._loss(p2, z1)) / 2

            loss_simsiam = tf.reduce_mean(loss_simsiam)
            loss_decay = sum(self.encoder.losses)

            loss = loss_simsiam + loss_decay
            total_loss = loss / self._num_workers

        trainable_vars = self.encoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        proj_std = tf.reduce_mean(tf.math.reduce_std(tf.math.l2_normalize(tf.concat((z1, z2), axis=0), axis=-1), axis=-1))
        pred_std = tf.reduce_mean(tf.math.reduce_std(tf.math.l2_normalize(tf.concat((p1, p2), axis=0), axis=-1), axis=-1))
        results = {
            'loss': loss, 
            'loss_simsiam': loss_simsiam, 
            'weight_decay': loss_decay, 
            'proj_std': proj_std,
            'pred_std': pred_std}
        return results