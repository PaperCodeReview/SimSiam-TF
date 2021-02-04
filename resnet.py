import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import Model

from layer import _conv2d
from layer import _batchnorm
from layer import _dense


def ResNet18(
    include_top=True,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs):
    
    norm = kwargs.pop('norm')
    DEFAULT_ARGS = {
        'use_bias': kwargs.pop('use_bias'), 
        'kernel_regularizer': kwargs.pop('kernel_regularizer')}

    def block0(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
        if conv_shortcut:
            shortcut = _conv2d(**DEFAULT_ARGS)(filters, 1, strides=stride, name=name+'_0_conv')(x)
            shortcut = _batchnorm(norm=norm)(epsilon=1.001e-5, name=name+'_0_bn')(shortcut)
        else:
            shortcut = x

        x = _conv2d(**DEFAULT_ARGS)(filters, kernel_size, strides=stride, padding='SAME', name=name+'_1_conv')(x)
        x = _batchnorm(norm=norm)(epsilon=1.001e-5, name=name+'_1_bn')(x)
        x = Activation('relu', name=name+'_1_relu')(x)

        x = _conv2d(**DEFAULT_ARGS)(filters, kernel_size, padding='SAME', name=name+'_2_conv')(x)
        x = _batchnorm(norm=norm)(epsilon=1.001e-5, name=name+'_2_bn')(x)

        x = Add(name=name+'_add')([shortcut, x])
        x = Activation('relu', name=name+'_out')(x)
        return x
        
    def stack0(x, filters, blocks, stride1=2, name=None):
        x = block0(x, filters, stride=stride1, name=name+'_block1')
        for i in range(2, blocks+1):
            x = block0(x, filters, conv_shortcut=False, name=name+'_block'+str(i))
        return x

    def stack_fn(x):
        x = stack0(x, 64, 2, stride1=1, name='conv2')
        x = stack0(x, 128, 2, name='conv3')
        x = stack0(x, 256, 2, name='conv4')
        return stack0(x, 512, 2, name='conv5')

    inputs = Input(shape=input_shape)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv1_pad')(inputs)
    x = _conv2d(**DEFAULT_ARGS)(64, 3, name='conv1_conv')(x)
    x = _batchnorm(norm=norm)(epsilon=1.001e-5, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)

    x = stack_fn(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = _dense(**DEFAULT_ARGS)(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)

    model = Model(inputs, x, name='resnet18')
    return model