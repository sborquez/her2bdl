"""
Uncertainty Models
==================

Models for Image Classification with  predictive distributions and uncertainty measure.

[1] GAL, Yarin. Uncertainty in deep learning. University of Cambridge, 2016, vol. 1, no 3.

[2] KENDALL, Alex; GAL, Yarin. What uncertainties do we need in bayesian deep learning 
    for computer vision?. arXiv preprint arXiv:1703.04977, 2017.

"""
#%%
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import (
    Input, Activation,
    Concatenate, Flatten, Dense, Lambda,
    Conv2D, MaxPooling2D, DepthwiseConv2D,
    BatchNormalization, Dropout
)
from .uncertainty import predictive_entropy, mutual_information, variation_ratio


__all__ = [
    'MCDropoutModel', 'AleatoricModel',
    'SimpleClassifierMCDropout', 'EfficientNetMCDropout',
    'HEDConvClassifierMCDropout', 'RGBConvClassifierMCDropout'
]

class MCDropoutModel(tf.keras.Model):
    """
    MonteCarlo Dropout base model. 
    
    MCDropoutModel is a abstract model for building new classification models
    and measure epistemic uncertainty using the method Monte-Carlo Dropout, defined
    by the author in [1].

    This implementation has methods from `tf.keras.Model` for training and predict,
    but also includes a method for GPU efficient stochastics forward passes, 
    predictive distribution and uncertainty estimation. 

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
        self.aleatoric_model = None
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

    def get_aleatoric_model(self):
        self.aleatoric_model = self.aleatoric_model or AleatoricModel(self)
        return self.aleatoric_model

    @staticmethod
    def build_encoder_model(input_shape, **kwargs):
        raise NotImplementedError

    @staticmethod
    def build_classifier_model(latent_variables_shape, num_classes, mc_dropout_rate=0.5, **kwargs):
        raise NotImplementedError

    def predict_with_epistemic_uncertainty(self, dataset, include_data=False, verbose=0, **kwargs):
        """
        Prediction and epistemic uncertainty results for x dataset.
        Parameters
        ----------
        dataset : `np.ndarray`  (batch_size, *input_shape)
            Data generator.
        include_data    : `bool``
            Returns x and y_true from dataset.
        kwargs : 
            keras.Model.predict kwargs.
        Return
        ------
            (`dict`, `dict`)
                Prediction and uncertainty results for x dataset.
        """
        predictions_results = None
        uncertainty_results = None
        pbar = tqdm(range(len(dataset))) if verbose >= 0 else range(len(dataset))
        for i in pbar:
            (X_batch, y_batch) = dataset[i]
            y_true_batch = y_batch.argmax(axis=1)
            predictions_batch = self.predict_distribution(
                x=X_batch, verbose=verbose, **kwargs
            )
            y_predictive_distribution_batch, y_pred_batch, y_predictions_samples_batch = predictions_batch
            uncertainty_batch = self.uncertainty(
                y_predictive_distribution=y_predictive_distribution_batch,
                y_predictions_samples=y_predictions_samples_batch,
                return_dict=True
            )
            if i == 0:
                predictions_results = {
                    "y_pred": y_pred_batch,
                    "y_predictions_samples":y_predictions_samples_batch,
                    "y_predictive_distribution": y_predictive_distribution_batch
                }
                uncertainty_results = uncertainty_batch
                if include_data:
                    data = {
                        "X": X_batch,
                        "y_true": y_true_batch
                    }
            else:
                predictions_results = {
                    "y_pred": np.hstack((predictions_results["y_pred"], y_pred_batch)),
                    "y_predictions_samples": np.vstack((predictions_results["y_predictions_samples"], y_predictions_samples_batch)),
                    "y_predictive_distribution": np.vstack((predictions_results["y_predictive_distribution"], y_predictive_distribution_batch))
                }
                uncertainty_results = {
                    k: np.hstack((uncertainty_results[k], v)) 
                    for (k,v) in uncertainty_batch.items()
                }
                if include_data:
                    data = {
                        "X": np.vstack((data["X"], X_batch)),
                        "y_true": np.hstack((data["y_true"], y_true_batch)),
                    }
        if include_data: return data, predictions_results, uncertainty_results
        return predictions_results, uncertainty_results

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
        # TODO: Use RepeatVector
        y_predictions_samples = np.array([
            self.classifier.predict(
                np.tile(z_i, (T, 1)),
                batch_size=self.mc_dropout_batch_size,
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
        ## Mutual Information
        I = mutual_information(y_predictive_distribution, y_predictions_samples)
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
        VR = variation_ratio(y_predictions_samples)
        return VR

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
        # Predictive Entropy
        H = predictive_entropy(y_predictive_distribution)
        return H

    def uncertainty(self, x=None, 
                    y_predictive_distribution=None, y_predictions_samples=None, 
                    get_predictive_entropy=None, get_multual_information=None,
                    get_variation_ratio=None, sample_size=None, verbose=0, return_dict=False,
                    **kwargs):
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
        if return_dict: return uncertainty
        return pd.DataFrame(uncertainty)


class AleatoricModel(tf.keras.Model):
    """
    Aleatoric Data Modeling. 
    
    AleatoricModel has basics methods from `tf.keras.Model` for training, the 
    model architecture is defined from a predefined MCDropoutModel.

    This model is used to evaluate the aleatoric uncertainty from data as explained
    in by the author in [2]. As this model requires a MCDropoutModel, it can use
    the trained weights of the Encoder submodel, but as the MC dropout layers
    are not required and it needs to learn the variance, Classifier's weights
    are ignored.

    Implemntation inspired by:
    https://towardsdatascience.com/building-a-bayesian-deep-learning-classifier-ece1845bc09
    https://github.com/ShellingFord221/My-implementation-of-What-Uncertainties-Do-We-Need-in-Bayesian-Deep-Learning-for-Computer-Vision/blob/master/classification_aleatoric.py
    """
    def __init__(self, mc_model):
        super(AleatoricModel, self).__init__()
        # Model parameters
        #self.mc_dropout_rate = mc_model.mc_dropout_rate
        self.image_shape = mc_model.image_shape
        self.batch_dim  = mc_model.batch_dim
        self.num_classes = mc_model.num_classes
        # Predictive distribution parameters
        self.sample_size = mc_model.sample_size
        self.encoder = None
        self.aleatoric_classifier = None
        self.build_from_mcmodel(mc_model)

    def build_from_mcmodel(self, mc_model):
        """
        Build a new aleatoric model using the architecture defined for `mc_model`.
        The encoder submodel is cloned and reused the same weights. The new
        aleatoric classifier reuse the same architecture from the `mc_model` classifier
        but removing the mc dropout layers, in consecuence, new weights are 
        initialized.

        Parameters
        ----------
        mc_model : `her2bdl.models.MCDropoutModel` 
            Base model to copy its architecture.
        """
        encoder, classifier = mc_model.layers
        # Clone Encoder Deterministic Model
        encoder_copy =  clone_model(encoder)
        encoder_copy.set_weights(encoder.get_weights())
        self.encoder = encoder_copy
        # Build Aleatoric Classifier Model
        layers = list(classifier.layers)
        input_layer = layers.pop(0)
        x = input_layer.output
        ## Copy architecture except classifier head and mc dropouts
        for layer in layers[:-1]:
            config = layer.get_config()
            if isinstance(layer, Dense) and (mc_model.mc_dropout_rate > 0):
                config["units"] = int(mc_model.mc_dropout_rate * config["units"])
            elif isinstance(layer, Dropout) and (mc_model.mc_dropout_rate > 0):
                continue
            x = type(layer)(**config)(x)
        ## New Aleatoric head layers
        logits = Dense(self.num_classes, name="head_logits_classifier")(x)
        variance = Dense(1, activation="softplus", name='variance')(x)
        logits_variance = Concatenate(name='logits_variance')([logits, variance])
        aleatoric_classifier =  tf.keras.Model(
            input_layer.output, logits_variance,
            name="aleatoric_classifier"
        )
        self.aleatoric_classifier = aleatoric_classifier

    def call(self, inputs):
        # Encode and extract features from input
        z = self.encoder(inputs)
        # Classify 
        return self.aleatoric_classifier(z)

    def build_aleatoric_loss(self, T=None):
        """
        Build the Negative Log Likehood loss with Monte-Carlo Sampling from [2].
        
        Parameters
        ----------
         T : `int` or `None` 
            Number of Monte-Carlo samples, if is `None`, it use models `sample_size`.
    	Return
        ------
            `function` with arguments (y_true, y_pred)
            Negative Log Likehood loss
        """
        T = T or self.sample_size
        num_classes = self.num_classes
        eps = tf.keras.backend.epsilon()
        def neg_log_likehood(y_true, y_pred):
            # Split prediction into variance and logits
            y_pred_std = tf.math.sqrt(y_pred[:,-1])
            y_pred_logits = y_pred[:,:-1]
            # Get Monte-Carlo samples and corrupt logits
            sample = tf.transpose(
                tfp.distributions.Normal(loc=tf.zeros_like(y_pred_std), scale=y_pred_std)
                .sample((T, num_classes))
            )
            x_t = tf.expand_dims(y_pred_logits, -1) + sample
            # Transform to probabilities
            batch_softmax_samples = tf.exp(x_t - tf.reduce_logsumexp(x_t, axis=1, keepdims=True))
            # Monte-Carlo Integration of softmax samples
            batch_softmax = tf.reduce_mean(batch_softmax_samples, 2, keepdims=True)
            # Negative Log Likehood
            loss_x = -1*tf.math.log(eps + tf.matmul(tf.expand_dims(y_true, 1), batch_softmax))
            return tf.squeeze(loss_x)
        return neg_log_likehood

    def predict_variance(self, x, return_y_pred=True, return_variance=True,
                          verbose=0, **kwargs):
        """
        Calculate predictive distrution and variance.
        See [2]
        This method returns the following `np.ndarray`:
        - `y_predictive_distribution` is a normalized histogram with shape:
          (batch_size, classes)
        - `y_pred` is the class predicted with shape: (batch_size,)
        - `y_predictions_variance` is the predictive variation,
          with shape: (batch_size,)
        Parameters
        ----------
        x : `np.ndarray`  (batch_size, *input_shape)
            Batch of inputs, if is one input, automatically it's converted to
            a batch.
        return_y_pred : `bool`
            Return argmax y_predictive_distribution. If is `False`
            `y_pred` is `None`.
        return_variance : `bool`
            Return predictive variation.
            If is `False` `y_predictions_variance` is `None`.
        kwargs : 
            keras.Model.predict kwargs.
        Return
        ------
            `tuple` of `np.ndarray` with length batch_size.
                Return 3 arrays with the model's predictive distributions.
        """
        if x.ndim == 3: x = np.array([x])
        assert x.ndim == 4, "Invalid x dimensions."
        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
            del kwargs["batch_size"]
        else:
            batch_size = 32
        logits_and_var_output = self.predict(x,
            batch_size=batch_size, verbose=verbose,**kwargs
        )
        logits = logits_and_var_output[:,:-1]
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y_predictive_distribution = (e_x / e_x.sum(axis=1, keepdims=True))
        y_predictions_variance = logits_and_var_output[:,-1]
        # class predictions
        y_pred = None
        if return_y_pred:
            y_pred = y_predictive_distribution.argmax(axis=1)
        if not return_variance:
            y_predictions_variance = None
        return y_predictive_distribution, y_pred, y_predictions_variance


    def predict_with_aleatoric_uncertainty(self, dataset, include_data=False, verbose=0, **kwargs):
        """
        Prediction and aleatoric uncertainty results for x dataset.
        Parameters
        ----------
        dataset : `np.ndarray`  (batch_size, *input_shape)
            Data generator.
        include_data    : `bool``
            Returns x and y_true from dataset.
        kwargs : 
            keras.Model.predict kwargs.
        Return
        ------
            (`dict`, `dict`)
                Prediction and uncertainty results for x dataset.
        """
        predictions_results = None
        uncertainty_results = None
        pbar = tqdm(range(len(dataset))) if verbose >= 0 else range(len(dataset))
        for i in pbar:
            (X_batch, y_batch) = dataset[i]
            y_true_batch = y_batch.argmax(axis=1)
            predictions_batch = self.predict(
                x=X_batch, verbose=verbose, **kwargs
            )
            y_predictive_distribution_batch = predictions_batch[:,:-1]
            y_pred_batch = y_predictive_distribution_batch.argmax(axis=1)
            predictive_variance = predictions_batch[:,-1]
            if i == 0:
                predictions_results = {
                    "y_pred": y_pred_batch,
                    "y_predictive_distribution": y_predictive_distribution_batch
                }
                uncertainty_results = {
                    "predictive_variance": predictive_variance
                }
                if include_data:
                    data = {
                        "X": X_batch,
                        "y_true": y_true_batch
                    }
            else:
                predictions_results = {
                    "y_pred": np.hstack(
                        (predictions_results["y_pred"], y_pred_batch)
                    ),
                    "y_predictive_distribution": np.vstack((
                        predictions_results["y_predictive_distribution"], 
                        y_predictive_distribution_batch
                    ))
                }
                uncertainty_results = {
                    "predictive_variance": np.hstack((
                        uncertainty_results["predictive_variance"], 
                        predictive_variance
                    ))
                }
                if include_data:
                    data = {
                        "X": np.vstack((data["X"], X_batch)),
                        "y_true": np.hstack((data["y_true"], y_true_batch)),
                    }
        if include_data: return data, predictions_results, uncertainty_results
        return predictions_results, uncertainty_results

    def uncertainty(self, x=None, y_predictive_variance=None, 
                    get_std=False,
                    verbose=0, return_dict=False, **kwargs):
        assert not (x is None and  y_predictive_variance is None),\
            "Must have an input x or a prediction"
        # Get predictions
        if x is not None:
            prediction = self.predict_variance(
                x, 
                return_y_pred=False, return_variance=True, verbose=verbose,
                **kwargs
            )
            y_predictive_distribution, _, y_predictive_variance= prediction
        # Uncertainty metrics
        uncertainty = {}
        ## Predictive Variance
        uncertainty["predictive variance"] = y_predictive_variance
        ## STD
        if get_std:
            uncertainty["predictive std"] = np.sqrt(y_predictive_variance)

        # Return DataFrame, where each column is a uncertainty metric
        # and each row is a image
        if return_dict: return uncertainty
        return pd.DataFrame(uncertainty)


"""
Custom Models
-------------

User defined architecture inheriting from MCDropoutModel.

These new models required two methods:

- build_encoder_model(input_shape, **kwargs)
    Defining the deterministic encoder model architecture. This model is used for
    extract features from image input.

- build_classifier_model(latent_variables_shape, num_classes, mc_dropout_rate=0.5, **kwargs)
    Defining the classifier stochastic model architecture. This model is used for
    generate the class prediction, and require the use of dropout layes with 
    training set True. 
"""

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
        x = Dense(512, name="head_dense_1")(x)
        x = BatchNormalization(name="head_batchnorm_1")(x)
        x = Activation("relu", name="head_relu_1")(x)
        if mc_dropout_rate > 0:
            x = Dropout(mc_dropout_rate, name="head_mc_dropout_1")(x, training=True) # MC dropout
        ### layer set
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
                classifier_dense_layers=[128, 128, 128], dense_activation='relu',
                mc_dropout_rate=0.2, sample_size=200, mc_dropout_batch_size=32,
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
    def build_classifier_model(latent_variables_shape, num_classes, mc_dropout_rate=0.2, **kwargs):
        # Architecture hyperparameters
        classifier_dense_layers = kwargs.get("classifier_dense_layers", [128, 128, 128])
        dense_activation = kwargs.get("activation", 'relu')
        mc_dropout_rate = mc_dropout_rate or 0.0
        ## Input layer
        x = clasifier_input = Input(shape=latent_variables_shape)
        ## Dense layers
        for i, units in enumerate(classifier_dense_layers, start=1):
            x = Dense(units, name=f"head_dense_{i}")(x)
            x = BatchNormalization(name=f"head_batchnorm_{i}")(x)
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


    def __init__(self, input_shape, num_classes, mc_dropout_rate=0.2, 
                encoder_kernel_sizes=[3, 3, 3], conv_activation='swish', ignore_eosin=False,
                classifier_dense_layers=[128, 128, 128], dense_activation='swish',
                sample_size=200, mc_dropout_batch_size=32, multual_information=True, 
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
            activation=conv_activation,
            ignore_eosin=ignore_eosin
        )
        self.ignore_eosin = ignore_eosin
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
        ignore_eosin = kwargs.get("ignore_eosin", False)
        ## Input Layers
        x = encoder_input = Input(shape=input_shape)
        x = Separate_HED_stains(name='stain_separator', ignore_eosin=ignore_eosin)(x)
        x = Lambda(lambda x: (x*2) - 1, name='scaler')(x)
        # Depthwise Conv
        x = DepthwiseConv2D(
            kernel_size=5, depth_multiplier=8, strides=2, padding="valid", 
            use_bias=False,
            depthwise_initializer=HEDConvClassifierMCDropout.CONV_KERNEL_INITIALIZER,
            name='stain_depthwiseConv2D'
        )(x)
        x = BatchNormalization(name='depthwiseConv2D_batchnorm')(x)
        x = Activation(activation_fn, name='depthwiseConv2D_activation')(x)
        ## initialize the layers in the first (CONV => RELU) * 2 => POOL
        ### layer set
        filters_2 = 5
        for i, kernel_size in enumerate(encoder_kernel_sizes):
            x = Conv2D(
                filters=2**(filters_2), kernel_size=kernel_size,
                padding="same", name=f"block{i}_conv2d_a", use_bias=False,
                kernel_initializer=HEDConvClassifierMCDropout.CONV_KERNEL_INITIALIZER
            )(x)
            x = BatchNormalization(name=f"block{i}_batchnorm_a")(x)
            x = Activation(activation_fn, name=f"block{i}_activation_a")(x)
            x = Conv2D(
                filters=2**filters_2, kernel_size=kernel_size,
                padding="valid", name=f"block{i}_conv2d_b", use_bias=False,
                kernel_initializer=HEDConvClassifierMCDropout.CONV_KERNEL_INITIALIZER
            )(x)
            x = BatchNormalization(name=f"block{i}_batchnorm_b")(x)
            x = Activation(activation_fn, name=f"block{i}_activation_b")(x)
            x = MaxPooling2D(pool_size=(2, 2), name=f"block{i}_maxpool")(x)
            filters_2 += 1
        x = Conv2D( filters=2**(filters_2-2), kernel_size=1,
                    name=f"feature_reduction_conv2d_k1")(x)
        x = Activation(activation_fn, name=f"feature_reduction_activation")(x)
        ## initialize the layers in our fully-connected layer sets
        ### layer set
        x = Flatten(name="head_flatten")(x)
        return tf.keras.Model(encoder_input, x, name="encoder")

    @staticmethod
    def build_classifier_model(latent_variables_shape, num_classes, mc_dropout_rate=0.2, **kwargs):
        # Architecture hyperparameters
        classifier_dense_layers = kwargs.get("classifier_dense_layers", [128, 128, 128])
        activation_fn = kwargs.get("activation", 'relu')
        mc_dropout_rate = mc_dropout_rate or 0.0
        ## Input layer
        x = clasifier_input = Input(shape=latent_variables_shape)
        ## Dense layers
        for i, units in enumerate(classifier_dense_layers, start=1):
            x = Dense(units, name=f"head_dense_{i}")(x)
            x = BatchNormalization(name=f"head_batchnorm_{i}")(x)
            x = Activation(activation_fn, name=f"head_activation_{i}")(x)
            if mc_dropout_rate > 0:
                x = Dropout(mc_dropout_rate, name=f"head_mc_dropout_{i}")(x, training=True) # MC dropout
        ## classsifier
        x = Dense(num_classes, activation="softmax", name="head_classifier")(x)
        ## Model
        classifier_model = tf.keras.Model(clasifier_input, x, name="classifier")
        return classifier_model

class RGBConvClassifierMCDropout(HEDConvClassifierMCDropout):
    """
    RGB Classifier with MonteCarlo Dropout.

    Convolutional neural network model with an architecture comparable to 
    the HEDConvClassifierMCDropout model.

    This model takes input images of any shape, and the input data
    should range [0, 255].
    
    """

    @staticmethod
    def build_encoder_model(input_shape, **kwargs):
        # Architecture hyperparameters
        activation_fn = kwargs.get("activation", 'relu')
        encoder_kernel_sizes = kwargs.get("encoder_kernel_sizes", [3, 3, 3])
        ## Input Layers
        x = encoder_input = Input(shape=input_shape)
        x = Lambda(lambda x: 2*(x/255) - 1, name='scaler')(x)
        # Depthwise Conv
        x = DepthwiseConv2D(
            kernel_size=5, depth_multiplier=8, strides=2, padding="valid",
            use_bias=False,
            depthwise_initializer=HEDConvClassifierMCDropout.CONV_KERNEL_INITIALIZER,
            name='stain_depthwiseConv2D'
        )(x)
        x = BatchNormalization(name='depthwiseConv2D_batchnorm')(x)
        x = Activation(activation_fn, name='depthwiseConv2D_activation')(x)
        ## initialize the layers in the first (CONV => RELU) * 2 => POOL
        ### layer set
        filters_2 = 5
        for i, kernel_size in enumerate(encoder_kernel_sizes):
            x = Conv2D(
                filters=2**(filters_2), kernel_size=kernel_size,
                padding="same", name=f"block{i}_conv2d_a", use_bias=False,
                kernel_initializer=HEDConvClassifierMCDropout.CONV_KERNEL_INITIALIZER
            )(x)
            x = BatchNormalization(name=f"block{i}_batchnorm_a")(x)
            x = Activation(activation_fn, name=f"block{i}_activation_a")(x)
            x = Conv2D(
                filters=2**filters_2, kernel_size=kernel_size,
                padding="valid", name=f"block{i}_conv2d_b", use_bias=False,
                kernel_initializer=HEDConvClassifierMCDropout.CONV_KERNEL_INITIALIZER
            )(x)
            x = BatchNormalization(name=f"block{i}_batchnorm_b")(x)
            x = Activation(activation_fn, name=f"block{i}_activation_b")(x)
            x = MaxPooling2D(pool_size=(2, 2), name=f"block{i}_maxpool")(x)
            filters_2 += 1
        x = Conv2D( filters=2**(filters_2-2), kernel_size=1,
                    name=f"feature_reduction_conv2d_k1")(x)
        x = Activation(activation_fn, name=f"feature_reduction_activation")(x)            
        ## initialize the layers in our fully-connected layer sets
        ### layer set
        x = Flatten(name="head_flatten")(x)
        return tf.keras.Model(encoder_input, x, name="encoder")