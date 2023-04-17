# Conditional Generative Adversarial Nets
## Info
- **Author:** Mehdi Mirza, Simon Osindero.
- **Link:** https://arxiv.org/pdf/1411.1784.pdf
- **Public date:** Nov 6, 2014.
- **Dataset:** `MNIST`, `MIR Flickr 25,000`.
- **Tags**: `GANs`, `Conditional-GANs`, `Unsupervised learning`.
 
## Summary
There is no way to control the generated data in the unconditional model. However, the data generation process can be directed by conditioning the model, such as by adding information based on the class label or on some part of the data for inpainting. Following this idea, the authors propose conditional-GANs, a conditional version of generative adversarial networks.

In conditional GANs, both the generator and discriminator are conditioned on some extra information `y`, which is fed into those models as additional layer input.

Loss function:

$$min_G max_D V(D,G) = E_{x \sim p_{data}(x)}[log(D(x\mid y))] + E_{z \sim p_z(z)}[log(1-D(G(z\mid y)))]$$

Experiment:
- Generate MNIST digits based on class labels.
- Generate image tags based on image features.