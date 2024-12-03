---
layout: post
title: "[Series]Deep Genertive Model Series: 2.Diffusion Model"
date: '2024-11-30 00:00:00 +800'
tags: [diffusion, generative model]
categories: generative_series
permalink: /generative_series/diffusion_model/
math: true
bibliography: diffusion_model.bib
# media_subpath: assets/post_images/generative_series
---

In this chapter, we are going to dig into the original paper of Denoising Diffusion Probabilistic Models (DDPM)[^1]. DDPM represent a pivotal advancement in diffusion-based generative modeling. Introduced by Ho et al. (2020)[^1], DDPM formulates the generative process as a gradual denoising procedure, effectively reversing a forward diffusion process that incrementally adds Gaussian noise to data.

To simplify, the idea of DDPM is:
> "Trying to add the image into pure noise, and learn how to denoise back."

Like all deep generative models, DDPM creates a mapping from one distribution to another.

DDPM consists of two main components: the **forward diffusion process** and the **backward diffusion process**.

---

### **2.1 Forward Diffusion Process**

![Backward Process](/assets/post_images/generative_series/backward_process.png)
*Figure 1: Forward and backward diffusion process. Image Source: {% cite ho2020 %}.*

The forward diffusion process in DDPM is conceptualized as a discrete-time Markov chain that progressively corrupts data by adding Gaussian noise over $T$ time steps. Formally, given an initial data sample $x_0$ drawn from the data distribution $p_{\text{data}}(x_0)$, the forward process generates a sequence of latent variables $x_1, x_2, \ldots, x_T$ via:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I),
$$

This tells us that for the next time step $x_t$, we sample from $\mathcal{N}(\sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$. Using the reparameterization trick, we can rewrite it as:

$$
x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I),
$$

which means adding noise $\sqrt{\beta_t} \epsilon_t$ to the data. By recursively applying the above transition, $x_t$ can be expressed directly in terms of $x_0$:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I),
$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

---

### **2.2 Backward Diffusion Process**


The crux of DDPM lies in reversing the forward diffusion process to generate data. The generative process is modeled as:

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)),
$$

where $\mu_\theta$ and $\Sigma_\theta$ are neural network parameterized mean and covariance functions, respectively. For a given noisy input $x_t$, the model $\theta$ predicts $x_{t-1}$ by sampling from the predicted $\mu_\theta$ and $\Sigma_\theta$.
We aim to ensure the reverse process aligns with the true posterior $q(x_{t-1} | x_t, x_0)$. Through derivations based on Bayes’ rule and Gaussian properties, this posterior is shown to be:

$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I),
$$

where:

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t,
$$

and:

$$
\tilde{\beta}_t = \frac{\beta_t (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}.
$$

---

### **2.3 Training Objective**

Minimizing the difference between the predicted noise $\epsilon_\theta$ and the true noise $\epsilon$ is simple, but it is **equivalent to minimizing the KL divergence between two distributions**. Let’s show why.

Rewriting $x_0$ in terms of $x_t$ and $\epsilon$:

$$
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}.
$$

Instead of directly predicting $x_0$, the model learns to predict the noise $\epsilon_\theta(x_t, t)$ using a neural network. The loss function for training becomes:

$$
\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right].
$$

The reverse process is parameterized as:

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)),
$$

while the true posterior is:

$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I).
$$

By parameterizing $\mu_\theta(x_t, t)$ in terms of $\epsilon_\theta(x_t, t)$:

$$
\mu_\theta(x_t, t) = \frac{x_t}{\sqrt{\bar{\alpha}_t}} - \frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}} \epsilon_\theta(x_t, t),
$$

minimizing the noise prediction error $\| \epsilon - \epsilon_\theta(x_t, t) \|^2$ is equivalent to minimizing the KL divergence. This establishes that the model learns the true data distribution by learning to denoise.

---

### **2.4 The Hidden Meaning of $\epsilon$**

Does $\epsilon$ have a deeper meaning? Yes. It is directly tied to the **score function**, which is the gradient of the log-probability of the data distribution.
For the Gaussian prior $q(x_t | x_0)$, the score function is:

$$
s_t(x_t) = \nabla_{x_t} \log q(x_t | x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1 - \bar{\alpha}_t}.
$$

Substituting $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$, we have:

$$
s_t(x_t) = -\frac{\sqrt{1 - \bar{\alpha}_t} \epsilon}{1 - \bar{\alpha}_t}.
$$

Thus, the score function is proportional to $-\epsilon$. Predicting $\epsilon$ during training is equivalent to estimating the score function $s_t(x_t)$, making $\epsilon$ the bridge between denoising and learning the geometry of the data distribution.


## 2.5 **What’s Next?**

  In the next chapter, we’ll explore how DDPM connects to stochastic differential equations (SDEs), providing a continuous-time perspective on diffusion-based generative modeling.

---

### **Appendix: Derivation of the True Posterior**

1. **Joint Distribution**  
   From the forward process:
   $$
   q(x_t, x_{t-1} | x_0) = q(x_t | x_{t-1}) q(x_{t-1} | x_0).
   $$

2. **Bayes' Rule**  
   Using Bayes’ rule:
   $$
   q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) q(x_{t-1} | x_0)}{q(x_t | x_0)}.
   $$

3. **Gaussian Product**  
   Both $q(x_t | x_{t-1})$ and $q(x_{t-1} | x_0)$ are Gaussians. The product of two Gaussians results in another Gaussian, where the mean and variance are derived by completing the square.

   - Mean:
     $$
     \tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t.
     $$

   - Variance:
     $$
     \tilde{\beta}_t = \frac{\beta_t (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}.
     $$
     

### Reference

<!-- [^1]: Ho et al. (2020), *Denoising Diffusion Probabilistic Models*. [Link](https://arxiv.org/abs/2006.11239) -->
<!-- ## References -->

{% bibliography --file diffusion_model.bib --cited %}