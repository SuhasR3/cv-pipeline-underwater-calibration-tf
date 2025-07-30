"""
Custom ResNet18 implementation for TensorFlow 2.x
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

class BasicBlock(tf.keras.layers.Layer):
    """Basic ResNet block with two 3x3 convolutions and a residual connection"""
    def __init__(self, filters, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=3, strides=stride,
                                   padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.downsample = downsample
        self.stride = stride

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def ResNet18(input_shape=(299, 299, 3), include_top=True, weights=None, classes=1000):
    """
    ResNet18 implementation for TensorFlow
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of the input images
    include_top : bool
        Whether to include the fully-connected layer at the top
    weights : str or None
        'imagenet' or None
    classes : int
        Number of classes (only used if include_top=True)
        
    Returns:
    --------
    model : tf.keras.Model
        ResNet18 model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial layers
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    
    # ResNet blocks
    # Layer 1
    for i in range(2):
        downsample = None
        if i == 0 and 64 != 64:
            downsample = Sequential([
                layers.Conv2D(64, kernel_size=1, strides=1, use_bias=False),
                layers.BatchNormalization()
            ])
        x = BasicBlock(64, stride=1 if i != 0 else 1, downsample=downsample)(x)
    
    # Layer 2
    for i in range(2):
        downsample = None
        if i == 0:
            downsample = Sequential([
                layers.Conv2D(128, kernel_size=1, strides=2, use_bias=False),
                layers.BatchNormalization()
            ])
        x = BasicBlock(128, stride=2 if i == 0 else 1, downsample=downsample)(x)
    
    # Layer 3
    for i in range(2):
        downsample = None
        if i == 0:
            downsample = Sequential([
                layers.Conv2D(256, kernel_size=1, strides=2, use_bias=False),
                layers.BatchNormalization()
            ])
        x = BasicBlock(256, stride=2 if i == 0 else 1, downsample=downsample)(x)
    
    # Layer 4
    for i in range(2):
        downsample = None
        if i == 0:
            downsample = Sequential([
                layers.Conv2D(512, kernel_size=1, strides=2, use_bias=False),
                layers.BatchNormalization()
            ])
        x = BasicBlock(512, stride=2 if i == 0 else 1, downsample=downsample)(x)
    
    # Final layers
    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(classes)(x)
    
    model = Model(inputs, x)
    
    # Load pretrained weights if specified
    if weights == 'imagenet':
        print("Warning: Pretrained ResNet18 weights for imagenet not available in this implementation")
        # In a real implementation, you would load pre-trained weights here
    
    return model