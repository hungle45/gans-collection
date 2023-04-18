# Image-to-Image Translation with Conditional Adversarial Networks
## Info
- **Author:** Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros.
- **Link:** https://arxiv.org/pdf/1611.07004.pdf
- **Public date:** Nov 26, 2018.
- **Dataset:**  `Cityscapes`, ` CMP Facades`,...
- **Tags**: `GANs`, `Unsupervised learning`.
 
## Summary
This paper investigates conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks learn a mapping from input image to output image as well as a loss function (the discriminator in GANs) to train this mapping.    

Network architecture:
- *Generator:* follow the “U-Net" shape with `Conv-BatchNorm-ReLU` in the encoder and `ConvTranspose-BatchNorm-ReLU` in the decoder.
- *Discriminator:* design `PatchGAN`, which tries to determine whether each $N*N$ patch in a image is real or fake. This makes the discriminator have fewer parameters and can be applied to any arbitrarily large image. The experiment results show that even when $N$ is much smaller than the image size, we still get high-quality results.

Loss function: has two parts
- The objective of a conditional GAN:
$$L_{cGAN}=E_{x,y}[logD(x,y)] + E_{x,z}[log(1-D(x,G(x,z)))]$$
- L1 distance, which encourages less blurring than L2:
$$L_{L1}(G)=E_{x,y,z}||y-G(x,z)||_1$$
$\rightarrow$ Final loss:
$$G=\text{min}_G \text{max}_D ~ L_{cGAN} + \lambda L_{L1}(G)$$

Experiments:
- ablation studies to isolate the effect of the L1 term, the GAN term and compare using a discriminator conditioned on the input with using an unconditional discriminator.
- compare the U-Net generator with the encoder-decoder generator.
- perform image translation on many tasks, such as synthesizing photos from label maps, reconstructing objects from edge maps, colorizing images, removing backgrounds, generating palettes, and some other tasks.