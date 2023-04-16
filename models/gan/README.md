# Generative Adversarial Nets
## Info
- **Author:** Ian J. Goodfellow, Jean Pouget-Abadie, MehdiMirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio.
- **Link:** https://arxiv.org/pdf/1406.2661.pdf
- **Public date:** Jun 10, 2014.
- **Dataset:** `MNIST`, `TFD`, `CIFAR-10`.
- **Tags**: `GANs`, `Unsupervised learning`.
 
## Summary
In the adversarial process, there are *two* models:

- *Generative model:* try to learn the generator's distribution $p_g$ over the data distribution $p_x$ from the given input noise $p_z$.
- *Discriminator:* try to distinguish which data sample comes from $p_g$ and which one comes from $p_x$.

Loss function:

$$min_G max_D V(D,G) = E_{x~p_x(x)}[log(D(x)] + E_{z~p_z(z)}[log(1-D(G(z))]$$

- To prevent $D$ from becoming overfit on the finite dataset, we can alternate between $k$ steps of optimizing $D$ and one step of optimizing $G$.
-  In early learning, $G$ is poor, so $D$ can easily reject samples that come from it. This leads $log(1-D(G(z))$ to saturate. To prevent this problem, we can train $G$ to maximize $log(D(G(z))$ instead.