# Kernel-stein-discrepancy-for-energy-based-model

## Key papers on Score-based Generative Modeling:

| Paper | Description |
| --- | --- |
| [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) |First score-based generative modeling: Denoising Score Matching + Annealed Langevin Dynamics|
|[How to Train Your Energy-Based Models](https://arxiv.org/abs/2101.03288)|A nice tutorial/introduction/summary on training EBM by Yang Song, including maximum likelihood, score matching, contrastive divergence and adversarial training.|
|[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)|A diffusion probabilistic models training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics. |
|[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)|A generative model framework based on SDE that encapsulates previous approaches in score-based generative modeling and diffusion probabilistic modeling, allowing for new sampling procedures and new modeling capabilities. In particular, we introduce a predictor-corrector framework to correct errors in the evolution of the discretized reverse-time SDE. |
|[On Maximum Likelihood Training of Score-Based Generative Models](https://arxiv.org/abs/2101.09258)|In this note, we show that such an objective is equivalent to maximum likelihood for certain choices of mixture weighting.|
|[Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)|We introduce a new family of deep neural network models. Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state using a neural network. The output of the network is computed using a black-box differential equation solver.|
|[Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)| Proposed the convolutional LSTM (ConvLSTM) and use it to build an end-to-end trainable model for the precipitation nowcasting problem.|
|[Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model](https://arxiv.org/abs/1706.03458)|Proposed trajectory GRU (TrajGRU) model that can actively learn the location-variant structure for recurrent connections. With the goal of making high-resolution forecasts.|

Kernelized Stein's Discrepancy:

| Paper | Description |
| --- | --- |
| [A Kernelized Stein Discrepancy for Goodness-of-fit Tests and Model Evaluation](https://arxiv.org/abs/1602.03253) |Kernelized Stein Discrepancy (KSD) as a computable measure of discrepancy between a sample of an unnormalized distribution|
|[Kernelized Complete Conditional Stein Discrepancy](https://arxiv.org/abs/1904.04478)|Conditional kernelized Stein discrepancy using joint kernel|
|[Learning the Stein Discrepancy for Training and Evaluating Energy-Based Models without Sampling](https://arxiv.org/abs/2002.05616)|Learning Stein discrepancy optimal evaluation function using neural network|
|[Minimum Stein Discrepancy Estimators](https://arxiv.org/abs/1906.08283)|Provide a unifying perspective of these techniques as minimum Stein discrepancy estimators, and use this lens to design new diffusion kernel Stein discrepancy (DKSD) and diffusion score matching (DSM) estimators with complementary strengths.|
|[Derivative reproducing properties for kernel methods in learning theory](https://core.ac.uk/download/pdf/82506111.pdf)|Deriavtives of functions in RKHS to hold reproducing properties|


Learning Energy based model (EBM):
| Paper | Description |
| --- | --- |
|[EBMs Trained with Maximum Likelihood are Generator Models Trained with a Self-adverserial Loss](https://arxiv.org/abs/2102.11757#:~:text=Maximum%20likelihood%20estimation%20is%20widely,algorithms%20such%20as%20Langevin%20dynamics.)|The authors show that reintroducing the noise in the dynamics does not lead to a qualitative change in the behavior, and merely reduces the quality of the generator. We thus show that EBM training is effectively a self-adversarial procedure rather than maximum likelihood estimation.|
|[On Maximum Likelihood Training of Score-Based Generative Models](https://arxiv.org/abs/2101.09258)|In this note, we show that such an objective is equivalent to maximum likelihood for certain choices of mixture weighting.|

Kernel Mean Embedding and Conditional Mean Embedding:

| Paper | Description |
| --- | --- |
|[Kernel Mean Embedding of Distributions: A Review and Beyond](https://arxiv.org/abs/1605.09522)|A review of kernel mean embedding and conditional kernel mean embedding for distribution|
|[Kernel dimension reduction in regression](https://arxiv.org/abs/0908.1854)|Kernel dimension reduction|
|[Kernel Conditional Density Operators](https://arxiv.org/abs/1905.11255)|Conditional Density estimation with kernel mean embedding method|
|[McGan: Mean and Covariance Feature Matching GAN](https://arxiv.org/abs/1702.08398)|Training Generative Adversarial Networks (GAN) based on matching statistics of distributions embedded in a finite dimensional feature space. Mean and covariance feature matching IPMs.|


Nonparametric Conditional Density Estimation
| Paper | Description |
| --- | --- |
|[Nonparametric Density Estimation for High-Dimensional Data - Algorithms and Applications](https://arxiv.org/pdf/1904.00176.pdf)|A review paper on nonparametric density estimation for high-dimensional data|
|[Conditional Density Estimation with Neural Networks: Best Practices and Benchmarks](https://arxiv.org/abs/1903.00954)|The paper develops best practices for conditional density estimation for finance applications with neural networks, grounded on mathematical insights and empirical evaluations. |
|[Nonparametric Conditional Density Estimation in a High-Dimensional Regression Setting](https://arxiv.org/abs/1604.00540)|Proposed a new nonparametric estimator of conditional density that adapts to sparse (low-dimensional) structure in x. |



# Conditional_VAE
## Related papers:

| Paper | Description |
| --- | --- |
| [Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models) | Conditional VAE |
| [Deep Generative Models with Learnable Knowledge Constraints](https://arxiv.org/pdf/1806.09764.pdf) | Learn generative models under constraints |
| [Semi-Supervised Learning with Deep Generative Models](https://arxiv.org/abs/1406.5298) | VAE as semi-supervised learning |
| [Projection Pursuit Regression](https://www.tandfonline.com/doi/abs/10.1080/01621459.1981.10477729) | Projection pursuit regression|
| [Variational Autoencoder with Arbitrary Conditioning](https://arxiv.org/abs/1806.02382) | Variational autoencoder that can be conditioned on an arbitrary subset of observed features and then sample the remaining features in "one shot". |
| [Infinite Variational Autoencoder for Semi-Supervised Learning](http://openaccess.thecvf.com/content_cvpr_2017/papers/Abbasnejad_Infinite_Variational_Autoencoder_CVPR_2017_paper.pdf) | Train a generative model using unlabelled data, and then use this model combined with labelled data to train a discriminative model for classification. |
|[Adversarial Autoencoders](https://arxiv.org/abs/1511.05644)|"adversarial autoencoder" (AAE), which is a probabilistic autoencoder that uses the recently proposed generative adversarial networks (GAN) to perform variational inference |
|[Adversarial Symmetric Variational Autoencoder](http://people.ee.duke.edu/~lcarin/AS_VAE.pdf)|Adversarial Symmetric Variational Autoencoder, using a adversial training in VAE, which has a very interesting comment on the Maximum likelihood estimation|


Semi-supervised learning with Deep Generative Models: [Pytorch implementation](https://github.com/wohlert/semi-supervised-pytorch)

Generative models (AAE included): [Pytorch implementation](https://github.com/wiseodd/generative-models)

