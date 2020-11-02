# Uncertainty Model for Image Classification

## Models

### Bayesian Deep Learning

### Sub-Models

The base model is build by the composition of two models: `encoder` and `classifier`.
The former, extract relevant features for a input image and generate the `latent_variables` vector. Is a **deterministic model**, i.e.  generate the same output for a given input.

<img src="https://render.githubusercontent.com/render/math?math=\Large z=\mathbf{E}(x, \omega^{(e)})">

The latter, use `latent_variables` as input for classify into `K`classes. The `classifier` is a  **stochastic model**, i.e., each forward pass can generate different outputs with the same input.

<!--$$p(y| z, \omega^{(c)}_t) = \mathbf{C}(z, \omega^{(c)}_t)$$-->
<img src="https://render.githubusercontent.com/render/math?math=\Large p(y| z, \omega^{(c)}_t) = \mathbf{C}(z, \omega^{(c)}_t)">

<!--$$\hat{p}(y|z) \approx \frac{1}{T}\sum_t p(y=C_k| z, \omega_t^{(c)})$$-->
<img src="https://render.githubusercontent.com/render/math?math=\Large \hat{p}(y|z) \approx \frac{1}{T}\sum_t p(y=C_k| z, \omega_t^{(c)})">

<!--$$\hat{y}_t = \argmax_{k} \hat{p}(y=C_k|z)$$-->
<img src="https://render.githubusercontent.com/render/math?math=\Large \hat{y}_t = \argmax_{k} \hat{p}(y=C_k|z)">

Where <img src="https://render.githubusercontent.com/render/math?math=\hat{p}(y|z)"> is the _predictive distibution_ the subindice <img src="https://render.githubusercontent.com/render/math?math=t"> indicate that is different for each <img src="https://render.githubusercontent.com/render/math?math=T"> forward passes.

The composition of theses models generate the Image Classifier Model <img src="https://render.githubusercontent.com/render/math?math=\mathbf{F}">:

<!--$$p(y|x, \hat{\omega}_t) = \mathbf{F}(x, \hat{\omega}_t) = \mathbf{C}(\mathbf{E}(x, \omega^{(e)}), \omega^{(c)}_t)$$-->
<img src="https://render.githubusercontent.com/render/math?math=\Large p(y|x, \hat{\omega}_t) = \mathbf{F}(x, \hat{\omega}_t) = \mathbf{C}(\mathbf{E}(x, \omega^{(e)}), \omega^{(c)}_t)">

<!--$$\hat{p}(y|x) \approx \frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t)$$-->
<img src="https://render.githubusercontent.com/render/math?math=\Large \hat{p}(y|x) \approx \frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t)">

<!--$$\hat{y}_t = \argmax_{k} \hat{p}(y=C_k|x)$$-->
<img src="https://render.githubusercontent.com/render/math?math=\Large \hat{y}_t = \argmax_{k} \hat{p}(y=C_k|x)">

Where <img src="https://render.githubusercontent.com/render/math?math=\hat{\omega}_t := \{\omega^{(e)}, \omega^{(c)}_t\}">. The model  <img src="https://render.githubusercontent.com/render/math?math=\mathbf{F}"> is summarized in the next figure:

![Base Model F](https://raw.githubusercontent.com/sborquez/her2bdl/uncertainty_models/her2bdl/models/images/assets/BaseModel.png)

### EfficientNet
This approach can reuse any well-known image classification architecture as encoder model <img src="https://render.githubusercontent.com/render/math?math=\mathbf{E}">, and fine-tuning its parameters.

For instance, this work use the `EfficientNet` as encoder model. See more. [[2](#[2])]

## Predictive Distribution


### Stochastics Forward Passes


The estimation of the **predictive distribution**  <img src="https://render.githubusercontent.com/render/math?math=\hat{p}(y|x)"> requiere of multiples forward passes with the same input <img src="https://render.githubusercontent.com/render/math?math=x">,which can be expensive.

Spliting the model in a *deterministic* and *stochastic* submodels, allows to reduce the time and resource required to compute the predictive distibution by reusing the deterministic latent variable <img src="https://render.githubusercontent.com/render/math?math=z"> and forward pass <img src="https://render.githubusercontent.com/render/math?math=\mathbf{E}(x)"> just once. 

Since the number of parameters <img src="https://render.githubusercontent.com/render/math?math=\mathbf{C}"> usually is smaller than <img src="https://render.githubusercontent.com/render/math?math=\mathbf{E}">, <img src="https://render.githubusercontent.com/render/math?math=T"> copies of <img src="https://render.githubusercontent.com/render/math?math=z"> can be stored in just one batch, thus a multiples forward passes of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{C}(z)"> can be computed by just one forward pass and a batch of size <img src="https://render.githubusercontent.com/render/math?math=T">. This runs faster due to the parallelization provided by the GPU.

## Metrics

### Confuision Matrix

### Uncertainty for Classification

Uncertainty metrics for classification models. See more. [[1](#[1])]

#### Predictive Entropy

Average amount of information contained in the predictive
distribution:

<!--$$\hat{\mathbb{H}}[y|x, D_{\text{train}}] := - \sum_{k} (\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t)) \log(\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t))$$-->
<img src="https://render.githubusercontent.com/render/math?math=\Large \hat{\mathbb{H}}[y|x, D_{\text{train}}] := - \sum_{k} (\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t)) \log(\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t))">

<!--$$(\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t))$$-->
Where <img src="https://render.githubusercontent.com/render/math?math=(\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t))"> is the **predictive distribution** for <img src="https://render.githubusercontent.com/render/math?math=T"> **stochastics forward pass**. Each pass correspond to a sample model <img src="https://render.githubusercontent.com/render/math?math=\hat{\omega}_t">. 

If <img src="https://render.githubusercontent.com/render/math?math=\T \rightarrow \infty">:

<!--$$\hat{\mathbb{H}}[y|x, D_{\text{train}}] \approx - \sum_{k} (\int p(y=C_k| x, \omega) d\omega) \log(\int p(y=C_k| x, \omega)d\omega)$$-->
<img src="https://render.githubusercontent.com/render/math?math=\Large \hat{\mathbb{H}}[y|x, D_{\text{train}}] \approx - \sum_{k} (\int p(y=C_k| x, \omega) d\omega) \log(\int p(y=C_k| x, \omega)d\omega)">

Where <img src="https://render.githubusercontent.com/render/math?math=(\int p(y=C_k| x, \omega) d\omega)"> is the  **"real" distribution**.

Predictive entropy is a biases estimator. The bias of this estimator will decrease as <img src="https://render.githubusercontent.com/render/math?math=T"> increases.


#### Mutual Information



#### Varition Rate

## References

### [1]
* GAL, Yarin. Uncertainty in deep learning. University of Cambridge, 2016, vol. 1, no 3.
### [2]
* TAN, Mingxing; LE, Quoc V. Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946, 2019.
