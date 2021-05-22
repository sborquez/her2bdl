"""
Uncertainty Meassures
=====================

Uncertainty measures for classification.

[1] GAL, Yarin. Uncertainty in deep learning. University of Cambridge, 2016, vol. 1, no 3.

"""
import numpy as np
from scipy import stats


__all__ = [
    'predictive_entropy', 'mutual_information', 'variation_ratio'
]

def predictive_entropy(y_predictive_distribution, is_sample=False, normalize=False):
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
    y_predictive_distribution : `np.ndarray` (batch_size, classes)
        Model's predictive distributions (normalized histogram).
    is_sample:  `bool`
        Calculate entropy for a sample instead of a batch of samples
    normalize:  `bool`
        Change range into [0,1]
    Return
    ------
        ``np.ndarray` with shape (batch_size,).
            Return predictive entropy for a batch.
    """
    # To batch format
    if is_sample: 
        y_predictive_distribution = np.expand_dims(y_predictive_distribution, 0)
    # Numerical Stability
    eps = np.finfo(y_predictive_distribution.dtype).tiny #.eps
    y_log_predictive_distribution = np.log(eps + y_predictive_distribution)
    # Predictive Entropy
    H = -1*np.sum(y_predictive_distribution * y_log_predictive_distribution, axis=1)
    if normalize:
        k = y_predictive_distribution.shape[1]
        H = H/np.log(k)
    return H[0] if is_sample else H


def mutual_information(y_predictive_distribution, y_predictions_samples, 
                      is_sample=False, normalize=False):
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
    y_predictive_distribution : `np.ndarray` (batch_size, classes)
        Model's predictive distributions (normalized histogram). 
    y_predictions_samples : `np.ndarray` (batch_size, sample size, classes)
        Forward pass samples.
    is_sample:  `bool`
        Calculate entropy for a sample instead of a batch of samples
    normalize:  `bool`
        Change range into [0,1]
    Return
    ------
        ``np.ndarray` with shape (batch_size,).
            Return mutual information for a batch.
    """
    # To batch format
    if is_sample:
        y_predictive_distribution = np.expand_dims(y_predictive_distribution, 0)
        y_predictions_samples = np.expand_dims(y_predictions_samples, 0)
    
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
    if normalize:
        k = y_predictive_distribution.shape[1]
        I = I/np.log(k)
    return I[0] if is_sample else I

def variation_ratio(y_predictions_samples, is_sample=False):
    """
    Collecting a set of $T$ labels $y_t$ from multiple stochastic forward
    passes on the same input, we can find the mode of the distribution 
    $k^∗ = \argmax_{k}\sum_t \mathbb{1}[y_t = C_k]$, and the number of
    times it was sampled $f_x = \sum_t\mathbb{1}[y_t = C_{k^*}]$.
    .. math::
    \text{variation-ratios}[x]=1-\frac{f_x}{T}
    Parameters
    ----------
    y_predictions_samples : `np.ndarray` (batch_size, classes)
        Model's predictive distributions (normalized histogram).
    Return
    ------
        ``np.ndarray` with shape (batch_size,).
            Return predictive entropy for a the batch.
    """
    # To batch format
    if is_sample:
        y_predictions_samples = np.expand_dims(y_predictions_samples, 0)
    batch_size, sample_size, num_classes = y_predictions_samples.shape
    # Sample a class for each forward pass
    cum_dist = y_predictions_samples.cumsum(axis=-1)
    Y_T = (np.random.rand(batch_size, sample_size, 1) < cum_dist).argmax(-1)
    # For each batch, get the frecuency of the mode.
    _, f = stats.mode(Y_T, axis=1)
    # variation-ratios
    T = sample_size
    variation_ratio_values = 1 - (f/T)
    VR = variation_ratio_values.flatten()
    return VR[0] if is_sample else VR