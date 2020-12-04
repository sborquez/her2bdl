"""
Uncertainty Models
==================

Models for Image Classification with  predictive distributions and uncertainty measure.

[1] GAL, Yarin. Uncertainty in deep learning. University of Cambridge, 2016, vol. 1, no 3.

"""
#%%
import numpy as np
from scipy import stats
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Activation,
    Flatten, Dense,
    Conv2D, MaxPooling2D, DepthwiseConv2D,
    BatchNormalization, Dropout
)


__all__ = [
    'MCDropoutModel',
    'SimpleClassifierMCDropout', 'EfficientNetMCDropout',
    'HEDConvClassifierMCDropout'
]

class MCDropoutModel(tf.keras.Model):
    """
    MonteCarlo Dropout base model. 
    
    MCDropoutModel has basics methods from `tf.keras.Model`, but also include 
    methods for GPU efficient stochastics forward passes, predictive distribution
    and uncertainty estimation.

    Develop new MonteCarlo Dropout models by inheriting from this class. These 
    new models must implement `build_encoder_model` and `build_classifier_model`
    and define their deterministic and stochastics corresponding to submodels
    meta-architecture.
    """
    def __init__(self, input_shape, num_classes, 
                mc_dropout_rate=0.5, sample_size=200, mc_dropout_batch_size=16, 
                multual_information=True, variation_ratio=True, 
                predictive_entropy=True):
        super(MCDropoutModel, self).__init__()
        # Model parameters
        self.mc_dropout_rate = mc_dropout_rate or 0.0
        self.image_shape = input_shape
        self.batch_dim  = 1 + len(self.image_shape) 
        self.num_classes = num_classes
        # Predictive distribution parameters
        self.sample_size = sample_size
        self.encoder = None
        self.classifier = None
        # Uncertainty measures
        self.get_multual_information = multual_information
        self.get_variation_ratio = variation_ratio
        self.get_predictive_entropy = predictive_entropy
        # batch size
        self.mc_dropout_batch_size = mc_dropout_batch_size
    
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

    def predict_distribution(self, x, return_y_pred=True, return_samples=True,
                             sample_size=None, verbose=0, **kwargs):
        """
        Calculate predictive distrution by T stochastic forward passes.
        See [1] 3.3.1 Uncertainty in classification
        This method returns the following `np.ndarray`:
        - `y_predictive_distribution` is a normalized histogram with shape:
          (batch_size, classes)
        - `y_pred` is the class predicted with shape: (batch_size,)
        - `y_predictions_samples` are samples from T forward passes samples,
          with shape: (batch_size, sample size, classes)
        Parameters
        ----------
        x : `np.ndarray`  (batch_size, *input_shape)
            Batch of inputs, if is one input, automatically it's converted to
            a batch.
        return_y_pred : `bool`
            Return argmax y_predictive_distribution. If is `False`
            `y_pred` is `None`.
        return_samples : `bool`
            Return forward passes samples. Required by variation_ratio  and 
            mutual information.
            If is `False` `y_predictions_samples` is `None`.
        sample_size    : `int` or `None`
            Number of fordward passes, also refers as `T`. If it is `None` 
            use model`s sample size.
        kwargs : 
            keras.Model.predict kwargs.
        Return
        ------
            `tuple` of `np.ndarray` with length batch_size.
                Return 3 arrays with the model's predictive distributions.
            If `return_samples` is `False` return only predictive distribution.
        """
        if x.ndim == 3: x = np.array([x])
        assert x.ndim == 4, "Invalid x dimensions."
        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
            del kwargs["batch_size"]
        else:
            batch_size = self.mc_dropout_batch_size
        T = sample_size or self.sample_size
        deterministic_output = self.encoder.predict(x,
            batch_size=batch_size, verbose=verbose,**kwargs
        )
        #deterministic_output_arr = deterministic_output.numpy()
        #del deterministic_output
        # T stochastics forward passes 
        y_predictions_samples = np.array([
            self.classifier.predict(
                np.tile(z_i, (T, 1)),
                batch_size=batch_size,
                verbose=verbose, **kwargs
            )
            for z_i in deterministic_output#_arr
        ])
        # predictive distribution
        y_predictive_distribution = y_predictions_samples.mean(axis=1)
        # class predictions
        y_pred = None
        if return_y_pred:
            y_pred = y_predictive_distribution.argmax(axis=1)
        if not return_samples:
            y_predictions_samples = None
        return y_predictive_distribution, y_pred, y_predictions_samples

    def mutual_information(self, x=None, y_predictive_distribution=None, 
                           y_predictions_samples=None, sample_size=None, 
                           **kwargs):
        """
        Measure of the mutual dependence between the two variables. More 
        specifically, it quantifies the "amount of information" (in units
        such as shannons, commonly called bits) obtained about one random
        variable through observing the other random variable.
        .. math::
        \mathbb{\tilde{I}}[y, \omega|x,  D_{\text{train}}] :=\
          -\sum_{k} (\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t))\
           \log(\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t))\
          + \frac{1}{T}\sum_{k}\sum_{t} p(y=C_k|x, \hat{\omega}_t)\
           \log(p(y=C_k|x, \hat{\omega}_t))
        Parameters
        ----------
        x : `np.ndarray`  (batch_size, *input_shape) or `None`
            Batch of inputs. If is `None` use precalculated `y_predictive_distribution`.
        y_predictive_distribution : `np.ndarray` (batch_size, classes) or `None`
            Model's predictive distributions (normalized histogram). Ignored
            if `x` is not `None`.
        y_predictions_samples : `np.ndarray` (batch_size, sample size, classes)
            Forward pass samples. Ignored if `x` is not `None`.
        sample_size    : `int` or `None`
            Number of fordward passes, also refers as `T`. If it is `None` 
            use model`s sample size.
        kwargs : 
            keras.Model.predict kwargs.
        Return
        ------
            ``np.ndarray` with shape (batch_size,).
                Return mutual information for a batch.
        """
        assert not (x is None and (y_predictive_distribution is None or y_predictions_samples is None) ),\
            "Must have an input x or a predictive distribution and predictions samples"
        if x is not None:
            sample_size = sample_size or self.sample_size
            prediction = self.predict_distribution(
                x, 
                return_y_pred=False, return_samples=True, 
                sample_size=sample_size, verbose=0, 
                **kwargs
            )
            y_predictive_distribution, _, y_predictions_samples = prediction
        sample_size = y_predictions_samples.shape[1]
        # Numerical Stability 
        eps = np.finfo(y_predictive_distribution.dtype).tiny #.eps        
        ## Entropy (batch, classes)
        y_log_predictive_distribution = np.log(eps + y_predictive_distribution) 
        H = -1*np.sum(y_predictive_distribution * y_log_predictive_distribution, axis=1)
        ## Expected value (batch, classes) 
        y_log_predictions_samples = np.log(eps + y_predictions_samples)
        minus_E = np.sum(y_predictions_samples*y_log_predictions_samples, axis=(1,2))
        minus_E /= sample_size
        ## Mutual Information
        I = H + minus_E
        return I

    def variation_ratio(self, x=None, y_predictions_samples=None, num_classes=None, sample_size=None, **kwargs):
        """
        Collecting a set of $T$ labels $y_t$ from multiple stochastic forward
        passes on the same input, we can find the mode of the distribution 
        $k^∗ = \argmax_{k}\sum_t \mathbb{1}[y_t = C_k]$, and the number of
        times it was sampled $f_x = \sum_t\mathbb{1}[y_t = C_{k^*}]$.
        .. math::
        \text{variation-ratios}[x]=1-\frac{f_x}{T}
        Parameters
        ----------
        x : `np.ndarray`  (batch_size, *input_shape) or `None`
            Batch of inputs. If is `None` use precalculated 
            `y_predictive_distribution`.
        y_predictions_samples : `np.ndarray` (batch_size, classes) or `None`
            Model's predictive distributions (normalized histogram). Ignore
            if `x` is not `None`.
        y_predictions_samples : `np.ndarray` (batch_size, sample size, classes)
            Forward pass samples. Ignored if `x` is not `None`.
        num_classes : `int` or `None`
            Number of classes.
        sample_size : `int` or `None`
            Number of fordward passes, also refers as `T`. If it is `None` 
            use model`s sample size.
        kwargs : 
            keras.Model.predict kwargs.
        Return
        ------
            ``np.ndarray` with shape (batch_size,).
                Return predictive entropy for a the batch.
        """
        assert not (x is None and y_predictions_samples is None),\
                "Must have an input x or predictions_samples"
        if x is not None:
            sample_size = sample_size or self.sample_size
            _, _, y_predictions_samples = self.predict_distribution(
                x, 
                return_y_pred=False, return_samples=True, 
                sample_size=sample_size, verbose=0, 
                **kwargs
            )
        batch_size, sample_size, num_classes = y_predictions_samples.shape
        # Sample a class for each forward pass
        cum_dist = y_predictions_samples.cumsum(axis=-1)
        Y_T = (np.random.rand(batch_size, sample_size, 1) < cum_dist).argmax(-1)
        # For each batch, get the frecuency of the mode.
        _, f = stats.mode(Y_T, axis=1)
        # variation-ratios
        T = sample_size
        variation_ratio_values = 1 - (f/T)
        return variation_ratio_values.flatten()

    def predictive_entropy(self, x=None, y_predictive_distribution=None, 
                            sample_size=None, **kwargs):
        """
        Predictive Entropy is the average amount of information contained in 
        the predictive distribution. Predictive entropy is a biases estimator.
        The bias of this estimator will decrease as $T$ (`sample_size`) increases.
        .. math::
        \mathbb{\tilde{H}}[y|x, D_{\text{train}}]  :=\
            - \sum_{k} (\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t)) \
                \log(\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t))
        Parameters
        ----------
        x : `np.ndarray`  (batch_size, *input_shape) or `None`
            Batch of inputs. If is `None` use precalculated `y_predictive_distribution`.
        y_predictive_distribution : `np.ndarray` (batch_size, classes) or `None`
            Model's predictive distributions (normalized histogram). Ignore
            if `x` is not `None`.
        sample_size    : `int` or `None`
            Number of fordward passes, also refers as `T`. If it is `None` 
            use model`s sample size.
        kwargs : 
            keras.Model.predict kwargs.
        Return
        ------
            ``np.ndarray` with shape (batch_size,).
                Return predictive entropy for a batch.
        """
        assert not (x is None and y_predictive_distribution is None),\
            "Must have an input x or a predictive distribution"
        if x is not None:
            sample_size = sample_size or self.sample_size
            y_predictive_distribution, _, _ = self.predict_distribution(
                x, 
                return_y_pred=False, return_samples=False, 
                sample_size=sample_size, verbose=0,
                **kwargs
            )
        # Numerical Stability 
        eps = np.finfo(y_predictive_distribution.dtype).tiny #.eps
        y_log_predictive_distribution = np.log(eps + y_predictive_distribution)
        # Predictive Entropy
        H = -1*np.sum(y_predictive_distribution * y_log_predictive_distribution, axis=1)
        return H

    def uncertainty(self, x=None, 
                    y_predictive_distribution=None, y_predictions_samples=None, 
                    get_predictive_entropy=None, get_multual_information=None,
                    get_variation_ratio=None, sample_size=None, verbose=0, **kwargs):
        assert not (x is None and ( y_predictive_distribution is None\
                                   and y_predictions_samples is None)),\
            "Must have an input x or a predictictions"
        # Get predictions
        if x is not None:
            sample_size = sample_size or self.sample_size
            prediction = self.predict_distribution(
                x, 
                return_y_pred=False, return_samples=True, 
                sample_size=sample_size, verbose=verbose,
                **kwargs
            )
            y_predictive_distribution, _, y_predictions_samples = prediction
        batch_size, sample_size, num_classes = y_predictions_samples.shape
        
        if verbose != 0: print("y_predictions_samples.shape:", batch_size, sample_size, num_classes)
        # Uncertainty metrics
        uncertainty = {}
        ## Predictive entropy
        get_predictive_entropy = get_predictive_entropy or self.get_predictive_entropy
        if get_predictive_entropy:
            if verbose != 0: print("predictive_entropy")
            H = self.predictive_entropy(
                y_predictive_distribution=y_predictive_distribution, 
                sample_size=sample_size, **kwargs
            )
            uncertainty["predictive entropy"] = H
        ## Mutual Information
        get_multual_information = get_multual_information or self.get_multual_information
        if get_multual_information:
            if verbose != 0: print("multual_information")
            I = self.mutual_information(
                y_predictive_distribution=y_predictive_distribution,
                y_predictions_samples=y_predictions_samples, 
                sample_size=sample_size, **kwargs)
            uncertainty["mutual information"] = I
        ## Variation Ratios
        get_variation_ratio = get_variation_ratio or self.get_variation_ratio
        if get_variation_ratio:
            if verbose != 0: print("variation_ratio")
            vr = self.variation_ratio(
                y_predictions_samples=y_predictions_samples, 
                num_classes=num_classes, sample_size=sample_size, **kwargs
            )
            uncertainty["variation-ratio"] = vr
        # Return DataFrame, where each column is a uncertainty metric
        # and each row is a image
        return pd.DataFrame(uncertainty)

class SimpleClassifierMCDropout(MCDropoutModel):
    """
    SimpleClassifier with MonteCarlo Dropout.

    Convolutional neural network model for testing purposes.
    """

    def __init__(self, input_shape, num_classes, mc_dropout_rate=0.5, 
                sample_size=200, mc_dropout_batch_size=16, multual_information=True, 
                variation_ratio=True, predictive_entropy=True):
        super(SimpleClassifierMCDropout, self).__init__(
            input_shape, num_classes, mc_dropout_rate, sample_size, 
            mc_dropout_batch_size, multual_information, variation_ratio, 
            predictive_entropy)
        # Architecture
        ## Encoder
        self.encoder = self.build_encoder_model(input_shape=input_shape)
        ## Stochastic Latent Variables
        latent_variables_shape = self.encoder.output.shape[1:]
        ## Classifier
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
        mc_dropout_rate = mc_dropout_rate or 0.0
        x = clasifier_input = Input(shape=latent_variables_shape)
        if mc_dropout_rate > 0:
            x = Dense(int(512/mc_dropout_rate), name="head_dense_1")(x)
        else:
            x = Dense(512, name="head_dense_1")(x)
        x = BatchNormalization(name="head_batchnorm_1")(x)
        x = Activation("relu", name="head_relu_1")(x)
        if mc_dropout_rate > 0:
            x = Dropout(mc_dropout_rate, name="head_mc_dropout_1")(x, training=True) # MC dropout
        ### layer set
        if mc_dropout_rate > 0:
            x = Dense(int(512/mc_dropout_rate), name="head_dense_2")(x)
        else:
            x = Dense(512, name="head_dense_2")(x)
        x = BatchNormalization(name="head_batchnorm_2")(x)
        x = Activation("relu", name="head_relu_2")(x)
        if mc_dropout_rate > 0:
            x = Dropout(mc_dropout_rate, name="head_mc_dropout_2")(x, training=True) # MC dropout
        ## initialize the layers in the softmax classifier layer set
        x = Dense(num_classes, activation="softmax", name="head_classifier")(x)
        return tf.keras.Model(clasifier_input, x, name="classifier")

#%%
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4,EfficientNetB5, EfficientNetB6, EfficientNetB7
)
class EfficientNetMCDropout(MCDropoutModel):
    """
    EfficientNet MonteCarlo Dropout. 
    Keras EfficientNets models wrappers with extra dropout layers.

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

    This model takes input images of shape (224, 224, 3), and the input data
    should range [0, 255]. Normalization is included as part of the model.
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

    def __init__(self, input_shape, num_classes, 
                base_model="B0", efficient_net_weights='imagenet',
                classifier_dense_layers=[256, 128], dense_activation='relu',
                mc_dropout_rate=0.5, sample_size=200, mc_dropout_batch_size=16,
                multual_information=True, variation_ratio=True,
                predictive_entropy=True):
        super(EfficientNetMCDropout, self).__init__(
            input_shape, num_classes, mc_dropout_rate, sample_size, 
            mc_dropout_batch_size, multual_information, variation_ratio,
            predictive_entropy
        )
        assert input_shape[:2] == EfficientNetMCDropout.__base_models_resolutions[base_model], "Input shape not supported by EfficientNetMCDropout"

        # Architecture
        ## Encoder - EfficientNet
        self.encoder = self.build_encoder_model(
            input_shape=input_shape, 
            base_model=base_model,
            efficient_net_weights=efficient_net_weights
        )
        ## Stochastic Latent Variables
        latent_variables_shape = self.encoder.output.shape[1:]
        ## Classifier
        self.classifier = self.build_classifier_model(
            latent_variables_shape=latent_variables_shape,
            num_classes=self.num_classes,
            mc_dropout_rate=self.mc_dropout_rate,
            classifier_dense_layers=classifier_dense_layers,
            activation=dense_activation
        )

    @staticmethod
    def build_encoder_model(input_shape, **kwargs):
        base_model = kwargs["base_model"]
        efficient_net_weights = kwargs["efficient_net_weights"]
        x = encoder_input = Input(shape=input_shape)
        # Efficient Net
        efficientBX = EfficientNetMCDropout.__base_models[base_model]
        x = efficientBX(include_top=False, weights=efficient_net_weights)(x)
        # Flatten layer
        x = Flatten(name="head_flatten")(x)
        # Model
        encoder_model = tf.keras.Model(encoder_input, x, name="encoder")
        ## fix weights
        if efficient_net_weights is not None:
            for layer in encoder_model.layers: layer.trainable = False
        return encoder_model

    @staticmethod
    def build_classifier_model(latent_variables_shape, num_classes, mc_dropout_rate=0.5, **kwargs):
        # Architecture hyperparameters
        classifier_dense_layers = kwargs.get("classifier_dense_layers", [256, 128])
        dense_activation = kwargs.get("activation", 'relu')
        mc_dropout_rate = mc_dropout_rate or 0.0
        ## Input layer
        x = clasifier_input = Input(shape=latent_variables_shape)
        ## Dense layers
        for i, units in enumerate(classifier_dense_layers, start=1):
            if mc_dropout_rate > 0:
                units = int(units/mc_dropout_rate)
            x = Dense(units, name=f"head_dense_{i}")(x)
            #x = BatchNormalization(name=f"head_batchnorm_{i}")(x)
            x = Activation(dense_activation, name=f"head_activation_{i}")(x)
            if mc_dropout_rate > 0:
                x = Dropout(mc_dropout_rate, name=f"head_mc_dropout_{i}")(x, training=True) # MC dropout
        ## classsifier
        x = Dense(num_classes, activation="softmax", name="head_classifier")(x)
        ## Model
        classifier_model = tf.keras.Model(clasifier_input, x, name="classifier")
        return classifier_model


from .layers import Separate_HED_stains

class HEDConvClassifierMCDropout(MCDropoutModel):
    """
    HED Stain separator Classifier with MonteCarlo Dropout.

    Convolutional neural network model that use the Separate_HED_stains layer.

    This model takes input images of any shape, and the input data
    should range [0, 255].
    
    """

    CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}


    def __init__(self, input_shape, num_classes, mc_dropout_rate=0.5, 
                encoder_kernel_sizes=[3, 3, 3], conv_activation='swish',
                classifier_dense_layers=[256, 128], dense_activation='swish',
                sample_size=200, mc_dropout_batch_size=16, multual_information=True, 
                variation_ratio=True, predictive_entropy=True):
        super(HEDConvClassifierMCDropout, self).__init__(
            input_shape, num_classes, mc_dropout_rate, sample_size, 
            mc_dropout_batch_size, multual_information, variation_ratio, 
            predictive_entropy)
        # Architecture
        ## Encoder
        self.encoder = self.build_encoder_model(
            input_shape=input_shape,
            encoder_kernel_sizes=encoder_kernel_sizes,
            activation=conv_activation
        )
        ## Stochastic Latent Variables
        latent_variables_shape = self.encoder.output.shape[1:]
        ## Classifier
        self.classifier = self.build_classifier_model(
            latent_variables_shape=latent_variables_shape,
            num_classes=self.num_classes,
            mc_dropout_rate=self.mc_dropout_rate, 
            classifier_dense_layers=classifier_dense_layers,
            activation=dense_activation
        )
    
    @staticmethod
    def build_encoder_model(input_shape, **kwargs):
        # Architecture hyperparameters
        activation_fn = kwargs.get("activation", 'relu')
        encoder_kernel_sizes = kwargs.get("encoder_kernel_sizes", [3, 3, 3])
        ## Input Layers
        x = encoder_input = Input(shape=input_shape)
        x = Separate_HED_stains(name='stain_separator')(x)
        # Delthwise Conv
        x = DepthwiseConv2D(
            kernel_size=3,
            depth_multiplier=8,
            strides=1,
            padding="valid",
            use_bias=False,
            depthwise_initializer=HEDConvClassifierMCDropout.CONV_KERNEL_INITIALIZER,
            name='stain_depthwiseConv2D'
        )(x)
        x = BatchNormalization(axis=3, name='depthwiseConv2D_batchnorm')(x)
        x = Activation(activation_fn, name='depthwiseConv2D_activation')(x)
        
        ## initialize the layers in the first (CONV => RELU) * 2 => POOL
        ### layer set\
        filters_2 = 4
        for i, kernel_size in enumerate(encoder_kernel_sizes):
            x = Conv2D(
                filters=2**filters_2, kernel_size=kernel_size,
                padding="valid", name=f"block{i}_conv2d_a", use_bias=False,
                kernel_initializer=HEDConvClassifierMCDropout.CONV_KERNEL_INITIALIZER
            )(x)
            x = BatchNormalization(axis=3, name=f"block{i}_batchnorm_a")(x)
            x = Activation(activation_fn, name=f"block{i}_activation_a")(x)
            x = Conv2D(
                filters=2**filters_2, kernel_size=kernel_size,
                padding="valid", name=f"block{i}_conv2d_b", use_bias=False,
                kernel_initializer=HEDConvClassifierMCDropout.CONV_KERNEL_INITIALIZER
            )(x)
            x = BatchNormalization(axis=3, name=f"block{i}_batchnorm_b")(x)
            x = Activation(activation_fn, name=f"block{i}_activation_b")(x)
            x = MaxPooling2D(pool_size=(2, 2), name=f"block{i}_maxpool")(x)
        ## initialize the layers in our fully-connected layer sets
        ### layer set
        x = Flatten(name="head_flatten")(x)
        return tf.keras.Model(encoder_input, x, name="encoder")

    @staticmethod
    def build_classifier_model(latent_variables_shape, num_classes, mc_dropout_rate=0.5, **kwargs):
        # Architecture hyperparameters
        classifier_dense_layers = kwargs.get("classifier_dense_layers", [256, 128])
        activation_fn = kwargs.get("activation", 'relu')
        mc_dropout_rate = mc_dropout_rate or 0.0
        ## Input layer
        x = clasifier_input = Input(shape=latent_variables_shape)
        ## Dense layers
        for i, units in enumerate(classifier_dense_layers, start=1):
            if mc_dropout_rate > 0:
                units = int(units/mc_dropout_rate)
            x = Dense(units, name=f"head_dense_{i}")(x)
            #x = BatchNormalization(name=f"head_batchnorm_{i}")(x)
            x = Activation(activation_fn, name=f"head_activation_{i}")(x)
            if mc_dropout_rate > 0:
                x = Dropout(mc_dropout_rate, name=f"head_mc_dropout_{i}")(x, training=True) # MC dropout
        ## classsifier
        x = Dense(num_classes, activation="softmax", name="head_classifier")(x)
        ## Model
        classifier_model = tf.keras.Model(clasifier_input, x, name="classifier")
        return classifier_model

