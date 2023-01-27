# Uncertainty Model for Image Classification

## Models

### Bayesian Deep Learning

$\mathbf{F}$ is a neural networks with parameters $\omega$ and the **softmax** output: 

$$p(y|x,\hat{\omega}) := [p(y = C_1|x,\hat{\omega}), \dots, p(y = C_K|x,\hat{\omega})] = \mathbf{F}(x, \hat{\omega})$$

The  **predictive distribution**, can be approximate diverses methods. MC-Dropout methods, require  repeating $T$ **stochastics forward passes**, by sampling $\hat{\omega}_t$ from learned weight distribution $q^*_0(\omega)$ (or $\Omega$) and evaluation $\mathbf{F}(x, \hat{\omega}_t)$.


$$p(y = C_k|x, \mathcal{D}_{\text{train}}) \approx \frac{1}{T}\sum_t p(y=C_k|x, \hat{w}_t)$$


$$p(y|x, \mathcal{D}_{\text{train}})  := [p(y = C_1|x, \mathcal{D}_{\text{train}}), \dots, p(y = C_K|x, \mathcal{D}_{\text{train}})]$$

Predictive entropy is a biases estimator. The bias of this estimator will decrease as $T$ increases.

### Sub-Models

The base model is build by the composition of two models: `encoder` and `classifier`.
The former, extract relevant features for a input image and generate the `latent_variables` vector. Is a **deterministic model**, i.e.  generate the same output for a given input.

$$z=\mathbf{E}(x, \omega^{(e)})$$

The latter, use `latent_variables` as input for classify into `K`classes. The `classifier` is a  **stochastic model**, i.e., each forward pass can generate different outputs with the same input.


$$p(y| z, \omega^{(c)}_t) = \mathbf{C}(z, \omega^{(c)}_t)$$

$$p(y=C_k|z, \mathcal{D}^{(z)}_{\text{train}}) \approx \frac{1}{T}\sum_t p(y=C_k| z, \omega_t^{(c)})$$

$$\hat{y} = \argmax_{k} p(y=C_k|z, \mathcal{D}^{(z)}_{\text{train}})$$

Where $\mathcal{D}^{(z)}_{\text{train}}$ is the train dataset in the latent variable space, and the subindice $t$ indicates that is different for each $T$ forward passes.

The composition of theses models generate the Image Classifier Model $\mathbf{F}$:

$$p(y|x, \hat{\omega}_t) = \mathbf{F}(x, \hat{\omega}_t) = \mathbf{C}(\mathbf{E}(x, \omega^{(e)}), \omega^{(c)}_t)$$

$$p(y = C_k|x, \mathcal{D}_{\text{train}}) \approx \frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t)$$

$$\hat{y} = \argmax_{k} p(y=C_k|x, \mathcal{D}_{\text{train}})$$

Where $\hat{\omega}_t := \{\omega^{(e)}, \omega^{(c)}_t\}$. The model  $\mathbf{F}$ is sumorized in the next figure:

![Base Model F](https://raw.githubusercontent.com/sborquez/her2bdl/uncertainty_models/her2bdl/models/images/assets/BaseModel.png)


### Stochastics Forward Passes in the GPU


The estimation of the **predictive distribution**  $p(y|x, \mathcal{D}_{\text{train}})$ requiere of multiples forward passes with the same input $x$, this can be memory expensive and time consuming.

Spliting the model in a *deterministic* and *stochastic* submodels, allows to reduce the time and resource required to compute the predictive distibution by reusing the deterministic latent variable $z$ and forward pass $\mathbf{E}(x)$ just once. 

Since the number of parameters $\mathbf{C}$ usually is smaller than $\mathbf{E}$, $T$ copies of $z$ can be stored in just one batch, thus a multiples forward passes of $\mathbf{C}(z)$ can be computed by just one forward pass and a batch of size $T$. This runs faster due to the parallelization provided by the **GPU**.

### EfficientNet

This approach can reuse any well-known image classification architecture as encoder model $\mathbf{E}$, and fine-tuning its parameters.

For instance, this work use the `EfficientNet` as encoder model. See more. [[2](#[2])]

## Metrics

### Confuision Matrix

### Uncertainty for Classification

Uncertainty metrics for classification models. See more. [[1](#[1])]

#### Predictive Entropy

Average amount of information contained in the predictive
distribution:

$$\begin{aligned}
\mathbb{H}[y|x, D_{\text{train}}] &:= -\sum_{k} p(y=C_k|x, D_{\text{train}}) \log(p(y=C_k|x, D_{\text{train}}))\\
&\approx - \sum_{k} (\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t)) \log(\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t))\\
&=\mathbb{\tilde{H}}[y|x, D_{\text{train}}] 
\end{aligned}$$

<!--
If $$\T \rightarrow \infty">:
$$\hat{\mathbb{H}}[y|x, D_{\text{train}}] \approx - \sum_{k} (\int p(y=C_k| x, \omega) d\omega) \log(\int p(y=C_k| x, \omega)d\omega)$$

Where $$(\int p(y=C_k| x, \omega) d\omega)"> is the  **"real" distribution**.-->




#### Mutual Information

Measure of the mutual dependence between the two variables. More specifically, it quantifies the "amount of information" (in units such as shannons, commonly called bits) obtained about one random variable through observing the other random variable.


$$\begin{aligned}
\mathbb{I}[y, \omega|x,  D_{\text{train}}] &:= \mathbb{H}[y|x, D_{\text{train}}] - \mathbb{E}_{p(\omega|D_{\text{train}})}[\mathbb{H}[y|x, \omega]]\\
&=-\sum_{k} p(y=C_k|x, D_{\text{train}}) \log(p(y=C_k|x, D_{\text{train}}))\\
&\qquad + \mathbb{E}_{p(\omega|D_{\text{train}})}[\sum_{k} p(y=C_k|x, \omega) \log(p(y=C_k|x, \omega))]\\
&\approx -\sum_{k} (\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t)) \log(\frac{1}{T}\sum_t p(y=C_k| x, \hat{\omega}_t))\\
&\qquad + \frac{1}{T}\sum_{k}\sum_{t} p(y=C_k|x, \hat{\omega}_t) \log(p(y=C_k|x, \hat{\omega}_t))\\
&=\mathbb{\tilde{I}}[y, \omega|x,  D_{\text{train}}]
\end{aligned}$$

#### Variation Ratios

To use variation ratios we would sample a label $y_t$ from the softmax probabilities at the end of each stochastic forward pass for a test input $x$. 

$$y_t \sim p(y|x,\hat{\omega}_t)$$

Collecting a set of $T$ labels $y_t$ from multiple stochastic forward passes on the same input, we can find the mode of the distribution $k^∗ = \argmax_{k}\sum_t \mathbb{1}[y_t = C_k]$, and the number of times it was sampled  $f_x = \sum_t\mathbb{1}[y_t = C_{k^*}]$.

$$\text{variation-ratios}[x]=1-\frac{f_x}{T}$$

The variation ratio is a measure of dispersion—how “spread” the distribution is around the mode.

## References

### [1]
* GAL, Yarin. Uncertainty in deep learning. University of Cambridge, 2016, vol. 1, no 3.
### [2]
* TAN, Mingxing; LE, Quoc V. Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946, 2019.
