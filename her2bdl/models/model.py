"""
Custom Models
=============

Models for Image Classification
"""
#%%
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Activation,
    Flatten, Dense,
    Conv2D, MaxPooling2D,
    BatchNormalization, Dropout
)


__all__ = ['SimpleClassifierMCDropout', 'EfficentNetMCDropout']

class ModelMCDropout(tf.keras.Model):
    def __init__(self, input_shape, num_classes, mc_dropout_rate=0.5, sample_size=500, multual_information=True, varition_ratio=True, predictive_entropy=True):
        super(ModelMCDropout, self).__init__()
        # Model parameters
        self.mc_dropout_rate = 0.0 if mc_dropout_rate is None else mc_dropout_rate
        self.image_shape = input_shape
        self.num_classes = num_classes
        # Predictive distribution parameters
        self.sample_size = sample_size

    # def build_encoder_model(self):
    #     raise NotImplementedError

    # def build_classifier_model(self):
    #     raise NotImplementedError

    def predict_distribution(self, x):
        pass

    def multual_information(self, x=None, predictive_distribution=None):
        assert not (x is None and predictive_distribution is None), "Must have an input x or a predictive distribution"

    def varition_ratio(self, x=None, predictive_distribution=None):
        assert not (x is None and predictive_distribution is None), "Must have an input x or a predictive distribution"
        

    def predictive_entropy(self, x=None, predictive_distribution=None):
        assert not (x is None and predictive_distribution is None), "Must have an input x or a predictive distribution"
        
    def uncertainty(self, x=None, predictive_distribution=None, 
                    multual_information=None, varition_ratio=None, predictive_entropy=None):
        assert not (x is None and predictive_distribution is None), "Must have an input x or a predictive distribution"


class SimpleClassifierMCDropout(ModelMCDropout):

    def __init__(self, input_shape, num_classes, mc_dropout_rate=0.5, sample_size=500, multual_information=True, varition_ratio=True, predictive_entropy=True):
        super(SimpleClassifierMCDropout, self).__init__(input_shape, num_classes, mc_dropout_rate, sample_size, multual_information, varition_ratio, predictive_entropy)
        # Architecture
        ## initialize the layers in the first (CONV => RELU) * 2 => POOL
        ### layer set
        self.conv1A = Conv2D(32, (3, 3), padding="same", name="block1_conv2d_a")
        self.bn1A = BatchNormalization(name="block1_batchnorm_a")
        self.act1A = Activation("relu", name="block1_relu_a")
        self.conv1B = Conv2D(32, (3, 3), padding="same", name="block1_conv2d_b")
        self.bn1B = BatchNormalization(name="block1_batchnorm_b")
        self.act1B = Activation("relu", name="block1_relu_b")
        self.pool1 = MaxPooling2D(pool_size=(2, 2), name="block1_maxpool")
        ## initialize the layers in the second (CONV => RELU) * 2 => POOL
        ### layer set
        self.conv2A = Conv2D(32, (3, 3), padding="same", name="block2_conv2d_a")
        self.bn2A = BatchNormalization(name="block2_batchnorm_a")
        self.act2A = Activation("relu", name="block2_relu_a")
        self.conv2B = Conv2D(32, (3, 3), padding="same", name="block2_conv2d_b")
        self.bn2B = BatchNormalization(name="block2_batchnorm_b")
        self.act2B = Activation("relu", name="block2_relu_b")
        self.pool2 = MaxPooling2D(pool_size=(2, 2), name="block2_maxpool")
        ## initialize the layers in our fully-connected layer sets
        ### layer set
        self.flatten = Flatten(name="head_flatten")
        self.dense3 = Dense(512, name="head_dense_1")
        self.bn3 = BatchNormalization(name="head_batchnorm_1")
        self.act3 = Activation("relu", name="head_relu_1")
        self.do3 = Dropout(self.mc_dropout_rate, name="head_mc_dropout_1")
        ### layer set
        self.dense4 = Dense(512, name="head_dense_2")
        self.bn4 = BatchNormalization(name="head_batchnorm_2")
        self.act4 = Activation("relu", name="head_relu_2")
        self.do4 = Dropout(self.mc_dropout_rate, name="head_mc_dropout_2")
        ## initialize the layers in the softmax classifier layer set
        self.classifier = Dense(self.num_classes, activation="softmax", name="head_classifier")

    def call(self, inputs):
        # build the first (CONV => RELU) * 2 => POOL layer set
        x = self.conv1A(inputs)
        x = self.bn1A(x)
        x = self.act1A(x)
        x = self.conv1B(x)
        x = self.bn1B(x)
        x = self.act1B(x)
        x = self.pool1(x)
        x = self.conv2A(x)
        x = self.bn2A(x)
        x = self.act2A(x)
        x = self.conv2B(x)
        x = self.bn2B(x)
        x = self.act2B(x)
        x = self.pool2(x)
        # build our FC layer set
        x = self.flatten(x)
        x = self.dense3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.do3(x, training=True) # MC dropout
        x = self.dense4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.do4(x, training=True) # MC dropout
        # build the softmax classifier
        x = self.classifier(x)
        # return the constructed model
        return x
#%%
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4,EfficientNetB5, EfficientNetB6, EfficientNetB7
)
class EfficentNetMCDropout(ModelMCDropout):
    """
    EfficientNet MonteCarlo Dropout. 
    Keras EfficentNets models wrappers with extra dropout layers.

    For B0 to B7 base models, the input shapes are different. Here is a list
    of input shape expected for each model:
    
    Base model  	resolution
    ==========================
    EfficientNetB0	224
    EfficientNetB1	240
    EfficientNetB2	260
    EfficientNetB3	300
    EfficientNetB4	380
    EfficientNetB5	456
    EfficientNetB6	528
    EfficientNetB7	600
    """

    __base_models_resolutions = {
        "B0": (224, 224),
        "B1": (240, 240),
        "B2": (260, 260),
        "B3": (300, 300),
        "B4": (380, 380),
        "B5": (456, 456),
        "B6": (528, 528),
        "B7": (600, 600)
    }

    __base_models = {
        "B0": EfficientNetB0,
        "B1": EfficientNetB1,
        "B2": EfficientNetB2,
        "B3": EfficientNetB3,
        "B4": EfficientNetB4,
        "B5": EfficientNetB5,
        "B6": EfficientNetB6,
        "B7": EfficientNetB7
    }

    def __init__(self, input_shape, num_classes, base_model = "B0", mc_dropout_rate=0.5, efficent_net_weights='imagenet', 
                sample_size=500, multual_information=True, varition_ratio=True, predictive_entropy=True):
        super(EfficentNetMCDropout, self).__init__(input_shape, num_classes, mc_dropout_rate, sample_size, multual_information, varition_ratio, predictive_entropy)
        assert input_shape[:2] == EfficentNetMCDropout.__base_models_resolutions[base_model], "Input shape not supported by EfficentNetMCDropout"

        # Architecture
        #self.input_tensor = Input(shape=input_shape)
        ## EfficentNet
        efficentBX = EfficentNetMCDropout.__base_models[base_model]
        self.efficentBX = efficentBX(
            include_top=False, weights=efficent_net_weights)
        ## initialize the layers in the softmax classifier layer set
        ### layer set
        self.flatten = Flatten(name="head_flatten")
        self.dense1 = Dense(512, name="head_dense_1")
        self.bn1 = BatchNormalization(name="head_batchnorm_1")
        self.act1 = Activation("relu", name="head_relu_1")
        if self.mc_dropout_rate > 0:
            self.do1 = Dropout(self.mc_dropout_rate, name="head_mc_dropout_1")
        ### layer set
        self.dense2 = Dense(512, name="head_dense_2")
        self.bn2 = BatchNormalization(name="head_batchnorm_2")
        self.act2 = Activation("relu", name="head_relu_2")
        if self.mc_dropout_rate > 0:
            self.do2 = Dropout(self.mc_dropout_rate, name="head_mc_dropout_2")
        ### classsifier
        self.classifier = Dense(self.num_classes, activation="softmax", name="head_classifier")

    def call(self, inputs):
        # EfficentNet
        if self.mc_dropout_rate > 0:
            #x = self.input_tensor(inputs)
            x = self.efficentBX(inputs)
            # build the softmax classifier
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.do1(x, training=True) # MC dropout
            x = self.dense2(x)
            x = self.bn2(x)
            x = self.act2(x)
            x = self.do2(x, training=True) # MC dropout
            x = self.classifier(x)
        else:
            #x = self.input_tensor(inputs)
            x = self.efficentBX(inputs)
            # build the softmax classifier
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.act1(x)
            x = self.bn1(x)
            x = self.dense2(x)
            x = self.act2(x)
            x = self.bn2(x)
            x = self.classifier(x)
        return x
