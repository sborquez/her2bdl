"""
Uncertainty Models
==================

Models for Image Classification with  predictive distributions and uncertainty measure.

[1] GAL, Yarin. Uncertainty in deep learning. University of Cambridge, 2016, vol. 1, no 3.

"""
#%%
import numpy as np
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
        self.batch_dim  = 1 + len(self.image_shape) 
        self.num_classes = num_classes
        # Predictive distribution parameters
        self.sample_size = sample_size
        self.encoder = None
        self.classifier = None
    
    def call(self, inputs):
        # Encode and extract features from input
        z = self.encoder(inputs)
        # Classify 
        return self.classifier(z)

    @staticmethod
    def build_encoder_model(input_shape, **kwargs):
        raise NotImplementedError

    @staticmethod
    def build_classifier_model(latent_variables_shape, num_classes, mc_dropout_rate=0.5, **kwargs):
        raise NotImplementedError

    def predict_distribution(self, x, return_y_pred=True, return_samples=True, sample_size=None, verbose=0, **kwargs):
        """
        Calculate predictive distrution by T stochastic forward passes.

        See [1] 3.3.1 Uncertainty in classification

        Parameters
        ----------
        x : `np.ndarray`  (batch_size, *input_shape)
            Batch of inputs.
        return_y_pred : `bool`
            Return argmax y_predictive_distribution. If is `False`
            `y_pred` is `None`.
        return_samples : `bool`
            Return forward pass samples. (required by varition_ratio)
            If is `False` `y_predictions_samples` is `None`.
        sample_size    : `int` or `None`
            Number of fordward passes, also refers as `T`. If it is `None` 
            use model`s sample size.
        kwargs : 
            keras.Model.predict kwargs.

        Return
        ------
            `list` of `np.ndarray` with length batch_size.
                Return 3 arrays with the model's predictive 
            distributions (normalized histogram), y class prediction and forward
            pass samples, `np.ndarray`with shape: (batch_size, sample size, classes). 
            If `return_samples` is `False` return only predictive distribution.
        """
        T = sample_size or self.sample_size
        deterministic_output = self.encoder.predict(x, verbose=verbose, **kwargs)
        # T stochastics forward passes 
        y_predictions_samples = np.array([
            self.classifier.predict(np.tile(z_i, (T, 1)), verbose=verbose, **kwargs)
            for z_i in deterministic_output
        ])
        # predictive distribution
        y_predictive_distribution = np.array([(1.0/T) * y_pred_T.sum(axis=0)  for y_pred_T in y_predictions_samples])

        # class predictions
        y_pred = None
        if return_y_pred:
            y_pred = y_predictive_distribution.argmax(axis=1)

        if not return_samples:
            y_predictions_samples = None

        return y_predictive_distribution, y_pred, y_predictions_samples

    def multual_information(self, x=None, y_predictive_distribution=None, sample_size=None, **kwargs):
        assert not (x is None and y_predictive_distribution is None),\
            "Must have an input x or a predictive distribution"
        if x is not None:
            sample_size = sample_size or self.sample_size
            _, y_predictive_distribution, _ = self.predict_distribution(
                x, 
                return_y_pred=False, return_samples=False, 
                sample_size=sample_size, verbose=0, 
                **kwargs
            )

    def varition_ratio(self, x=None, y_predictions_samples=None, **kwargs):
        assert not (x is None and y_predictions_samples is None),\
                "Must have an input x or predictions_samples"
        
    # def predictive_entropy(self, x=None, y_predictive_distribution=None, 
    #                        sample_size=None, **kwargs):
    #     """
    #     Average amount of information contained in the predictive distribution:
    #     Predictive entropy is a biases estimator. The bias of this estimator 
    #     will decrease as $T$ (`sample_size`) increases.
    #     .. math::
    #     \hat{\mathbb{H}}[y|x, D_{\text{train}}] := - \sum_{k} (\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t)) \log(\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t))
    #     Parameters
    #     ----------
    #     x : `np.ndarray`  (batch_size, *input_shape) or `None`
    #         Batch of inputs. If is `None` use precalculated `y_predictive_distribution`.
    #     y_predictive_distribution : `np.ndarray` (batch_size, classes) or `None`
    #         Model's predictive distributions (normalized histogram). Ignore
    #         if `x` is not `None`.
    #     sample_size    : `int` or `None`
    #         Number of fordward passes, also refers as `T`. If it is `None` 
    #         use model`s sample size.
    #     kwargs : 
    #         keras.Model.predict kwargs.
    #     Return
    #     ------
    #         ``np.ndarray` with length batch_size.
    #             Return predictive entropy for a the batch.
    #     """
    #     assert not (x is None and y_predictive_distribution is None),\
    #         "Must have an input x or a predictive distribution"
    #     if x is not None:
    #         sample_size = sample_size or self.sample_size
    #         _, y_predictive_distribution, _ = self.predict_distribution(
    #             x, 
    #             return_y_pred=False, return_samples=False, 
    #             sample_size=sample_size, verbose=0,
    #             **kwargs
    #         )
    #     # Numerical Stability 
    #     eps = np.finfo(y_predictive_distribution.dtype).tiny #.eps
    #     y_log_predictive_distribution = np.log(eps + y_predictive_distribution)
    #     # Predictive Entropy
    #     H = -1*np.sum(y_predictive_distribution * y_log_predictive_distribution, axis=1)
    #     return H

    def uncertainty(self, x=None, y_predictive_distribution=None, multual_information=None,
                    varition_ratio=None, predictive_entropy=None, **kwargs):
        assert not (x is None and y_predictive_distribution is None),\
            "Must have an input x or a predictive distribution"


class SimpleClassifierMCDropout(ModelMCDropout):

    def __init__(self, input_shape, num_classes, mc_dropout_rate=0.5, sample_size=500, multual_information=True, varition_ratio=True, predictive_entropy=True):
        super(SimpleClassifierMCDropout, self).__init__(input_shape, num_classes, mc_dropout_rate, sample_size, multual_information, varition_ratio, predictive_entropy)
        # Architecture
        # Encoder
        self.encoder = self.build_encoder_model(input_shape=input_shape)
        latent_variables_shape = self.encoder.output.shape[1:]
        self.classifier = self.build_classifier_model(
            latent_variables_shape=latent_variables_shape,
            mc_dropout_rate=self.mc_dropout_rate, 
            num_classes=self.num_classes
        )
    
    @staticmethod
    def build_encoder_model(input_shape, **kwargs):
        x = encoder_input = Input(shape=input_shape)
        ## initialize the layers in the first (CONV => RELU) * 2 => POOL
        ### layer set
        x = Conv2D(32, (3, 3), padding="same", name="block1_conv2d_a")(x)
        x = BatchNormalization(name="block1_batchnorm_a")(x)
        x = Activation("relu", name="block1_relu_a")(x)
        x = Conv2D(32, (3, 3), padding="same", name="block1_conv2d_b")(x)
        x = BatchNormalization(name="block1_batchnorm_b")(x)
        x = Activation("relu", name="block1_relu_b")(x)
        x = MaxPooling2D(pool_size=(2, 2), name="block1_maxpool")(x)
        ## initialize the layers in the second (CONV => RELU) * 2 => POOL
        ### layer set
        x = Conv2D(32, (3, 3), padding="same", name="block2_conv2d_a")(x)
        x = BatchNormalization(name="block2_batchnorm_a")(x)
        x = Activation("relu", name="block2_relu_a")(x)
        x = Conv2D(32, (3, 3), padding="same", name="block2_conv2d_b")(x)
        x = BatchNormalization(name="block2_batchnorm_b")(x)
        x = Activation("relu", name="block2_relu_b")(x)
        x = MaxPooling2D(pool_size=(2, 2), name="block2_maxpool")(x)
        ## initialize the layers in our fully-connected layer sets
        ### layer set
        x = Flatten(name="head_flatten")(x)
        return tf.keras.Model(encoder_input, x, name="encoder")

    @staticmethod
    def build_classifier_model(latent_variables_shape, num_classes, mc_dropout_rate=0.5, **kwargs):
        x = encoder_input = Input(shape=latent_variables_shape)
        x = Dense(int(512/mc_dropout_rate), name="head_dense_1")(x)
        x = BatchNormalization(name="head_batchnorm_1")(x)
        x = Activation("relu", name="head_relu_1")(x)
        x = Dropout(mc_dropout_rate, name="head_mc_dropout_1")(x, training=True) # MC dropout
        ### layer set
        x = Dense(int(512/mc_dropout_rate), name="head_dense_2")(x)
        x = BatchNormalization(name="head_batchnorm_2")(x)
        x = Activation("relu", name="head_relu_2")(x)
        x = Dropout(mc_dropout_rate, name="head_mc_dropout_2")(x, training=True) # MC dropout
        ## initialize the layers in the softmax classifier layer set
        x = Dense(num_classes, activation="softmax", name="head_classifier")(x)
        return tf.keras.Model(encoder_input, x, name="classifier")

#%%
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4,EfficientNetB5, EfficientNetB6, EfficientNetB7
)
class EfficentNetMCDropout(ModelMCDropout):
    #TODO: update to encode/classifier 
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
