# Uncertainty Model for Image Classification

## Models

### Sub-Models

### EfficientNet

See more. [[2](#[2])]

## Predictive Distribution

### Stochastics Forward Passes

#### GPU Optimized T forward passes

## Metrics

### Confuision Matrix

### Uncertainty for Classification

Uncertainty metrics for classification models. See more. [[1](#[1])]

#### Predictive Entropy

Average amount of information contained in the predictive
distribution:

$$\hat{\mathbb{H}}[y|x, D_{\text{train}}] := - \sum_{k} (\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t)) \log(\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t))$$

Where $(\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t))$ is the **predictive distribution** for $T$ **stochastics forward pass**. Each pass correspond to a sample model $\hat{\omega}_t$. 

If $T \rightarrow \infty$:

$$\hat{\mathbb{H}}[y|x, D_{\text{train}}] \approx - \sum_{k} (\int p(y=C_k| x, \omega) d\omega) \log(\int p(y=C_k| x, \omega)d\omega)$$

Where $(\int p(y=C_k| x, \omega) d\omega)$ is the  **"real" distribution**.

Predictive entropy is a biases estimator. The bias of this estimator will decrease as $T$ increases.


#### Mutual Information

#### Varition Rate

## References

### [1]
* GAL, Yarin. Uncertainty in deep learning. University of Cambridge, 2016, vol. 1, no 3.
### [2]
* TAN, Mingxing; LE, Quoc V. Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946, 2019.