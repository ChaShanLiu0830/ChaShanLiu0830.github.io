---
layout: post
title: "[Note]From Stochastic Processes to DDPMs"
date: '2024-08-19 00:00:00 +800'
tags: [diffusion, generative model]
math: true
# media_subpath: /_image/2024-05-04-[Implement]Improving Diffusion Inverse Problem Solving
# with Decoupled Noise Annealing/
---

## Introduction

In recent years, diffusion models have gained significant attention in the field of generative modeling, particularly due to their success in generating high-quality images. These models, inspired by the principles of thermodynamics and stochastic processes, offer a novel approach to data generation by progressively transforming noise into structured data through a carefully designed reverse diffusion process.

The foundation of these models lies in understanding the relationship between forward and backward diffusion processes. The forward process gradually corrupts data by adding noise, while the backward process aims to recover the original data by reversing this noise. This interplay between the two processes is crucial for building models that can generate realistic samples from complex data distributions.

In this post, we will dive into the mathematical framework underlying these diffusion processes, starting from the stochastic differential equations (SDEs) that describe them, and leading up to their application in Denoising Diffusion Probabilistic Models (DDPMs). By the end, you'll gain a deeper understanding of how these processes connect and how they form the backbone of modern generative models.

## From Forward Process to Backward Process


Starting with the diffusion process, we have:
$$
dX_t = f(X_t,t)\,dt + \sigma(t)\,dW_t
$$
Here, $f(X_t,t)$ represents the drift term, which determines the deterministic trend of the process, while $\sigma(t)$ is the diffusion coefficient, controlling the randomness or noise in the system. $W_t$ is the standard Wiener process, which is a mathematical model for random motion.

The probability distribution of the system evolves over time according to the Fokker-Planck equation, which is a partial differential equation describing the time evolution of the probability density function $p(x,t)$:

$$
\frac{\partial p(x,t)}{\partial t} = -\frac{\partial }{\partial x} \Big[f(x, t)\,p(x,t)\Big] + \frac{1}{2}\frac{\partial^2}{\partial x^2} \Big[\sigma^2(t)\,p(x,t)\Big]
$$

This equation consists of two terms on the right-hand side:
1. The first term represents the change in probability due to the drift, i.e., the deterministic part of the motion.
2. The second term accounts for the diffusion, representing the spread of probability due to the randomness in the process.

We can rewrite the Fokker-Planck equation in a more compact form by recognizing the term inside the derivative as the probability current $J(x,t)$:

$$
\frac{\partial p(x,t)}{\partial t} = -\frac{\partial }{\partial x} \Big[f(x, t)\,p(x,t) + \frac{1}{2}\frac{\partial}{\partial x}\Big[\sigma^2(t)\,p(x,t)\Big]\Big]
$$

Defining the probability current $J(x,t)$ as:

$$
J(x,t) = f(x, t)\,p(x,t) + \frac{1}{2}\frac{\partial}{\partial x}\Big[\sigma^2(t)\,p(x,t)\Big]
$$

the Fokker-Planck equation simplifies to:

$$
\frac{\partial p(x,t)}{\partial t} = -\frac{\partial J(x,t)}{\partial x}
$$

This is a well-known continuity equation, which expresses the conservation of probability: the rate of change of probability density at a point is equal to the net probability current flowing out of that point.

Now, consider the backward diffusion process, which describes the process when time is reversed. The SDE for the backward process is:

$$
d\tilde{X}_t = \tilde{f}(\tilde{X}_t,t)\,dt + \sigma(t)\,d\tilde{W}_t
$$

In this equation, $\tilde{f}(\tilde{X}_t,t)$ is the drift term for the time-reversed process, and $d\tilde{W}_t$ is a Wiener process in the reversed time direction. 

To account for time reversal, we need to reverse the sign of the first-order time derivative in the Fokker-Planck equation. This gives us:

$$
-\frac{\partial p(x,t)}{\partial t} = -\frac{\partial }{\partial x} \Big[\tilde{f}(x, t)\,p(x,t)\Big] + \frac{1}{2}\frac{\partial^2}{\partial x^2}\Big[\sigma^2(t)\,p(x,t)\Big]
$$

The backward process should evolve in such a way that it "undoes" the forward process. By adding the forward and backward equations and equating them, we obtain the relationship:

$$
\tilde{f}(x,t) = f(x,t) + \sigma^2(t) \frac{\partial \log p(x,t)}{\partial x}
$$

This equation shows that the drift in the backward process is not simply the negative of the forward drift; instead, it includes a correction term that depends on the gradient of the log-probability density. This correction ensures that the process evolves correctly when time is reversed.

Thus, the backward diffusion process is given by:

$$
d\tilde{X}_t = \left[ f(\tilde{X}_t,t) - \sigma^2(t)\nabla_{\tilde{X}_t} \log p(\tilde{X}_t,t) \right]\,dt + \sigma(t)\,d\tilde{W}_t
$$

This SDE describes how the process moves backward in time, ensuring that the probability distribution remains consistent with the original (forward) dynamics.

## From SDEs to DDPMs

Now that we have established the forward and backward diffusion processes, let's explore how this connects to Denoising Diffusion Probabilistic Models (DDPMs).

DDPMs can be expressed by the following SDE:

$$
dX_t = -\frac{1}{2} \beta(t)\,X_t\,dt + \sqrt{\beta(t)}\,dW_t
$$

Here, $\beta(t)$ is a time-dependent coefficient that controls the rate of diffusion and the strength of noise. This equation describes how the data (e.g., an image) is gradually corrupted by noise over time.

To connect this to the discrete-time setting often used in machine learning, we can discretize the continuous SDE. Consider $dX_t$ as $x_{t+1} - x_t$ and $dW_t$ as $\mathcal{N}(0,1)\sqrt{\Delta t}$, where $\mathcal{N}(0,1)$ represents a standard normal distribution. The equation becomes:

$$
x_{t+1} - x_t = -\frac{1}{2} \beta(t)\,x_t\,\Delta t + \sqrt{\beta(t)\Delta t}\,\mathcal{N}(0,1)
$$

Simplifying this, we get:

$$
\begin{aligned}[l]
x_{t+1} &= \left(1 - \frac{1}{2}\beta(t)\Delta t\right) x_t + \sqrt{\beta(t)\Delta t}\,\mathcal{N}(0,1) \\
&\approx \sqrt{1 - \beta(t)}\,x_t + \sqrt{\beta(t)}\,\mathcal{N}(0,1)
\end{aligned}
$$

This approximation shows how the process gradually corrupts the data by mixing it with noise. Each step $t$ brings the data closer to pure noise, which is the key idea behind the forward process in DDPMs.

The backward process in DDPMs, which is used for generating samples, is essentially the reverse of this process. By learning to reverse the noise corruption step by step, the model can generate data samples from pure noise, effectively denoising the noisy data.

This connection between SDEs and DDPMs illustrates how diffusion processes can be used to model complex data distributions, providing a powerful framework for generative modeling.