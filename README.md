# AutoEncoder

[AutoEncoder](https://en.wikipedia.org/wiki/Autoencoder#Regularized_autoencoders) is a type of Artificial Neural Networks. It is a data compression algorithm where the compression and decompression functions are 1) data-specific, 2) lossy, and 3) learned automatically.

![!Keras Sample](https://keras-cn.readthedocs.io/en/latest/legacy/images/autoencoder_schema.jpg)

# Implemented Models

<h3> Denoising AutoEncoder </h3>

Denoising autoencoders are an extension of the basic autoencoder, and represent a stochastic version of it. Denoising autoencoders attempt to address identity-function risk by randomly corrupting input (i.e. introducing noise) that the autoencoder must then reconstruct, or denoise. 

Reference: [pathmind](https://wiki.pathmind.com/denoising-autoencoder)

<h3> Variational AutoEncoder </h3>

Variational autoencoder models inherit autoencoder architecture, but make strong assumptions concerning the distribution of latent variables. They use variational approach for latent representation learning, which results in an additional loss component and specific training algorithm called Stochastic Gradient Variational Bayes (SGVB).

Reference: [pathmind](https://wiki.pathmind.com/variational-autoencoder)