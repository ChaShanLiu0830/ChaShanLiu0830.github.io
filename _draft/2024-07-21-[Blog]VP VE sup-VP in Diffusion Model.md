---
layout: post
title: "[Blog]VP, VE and sup-VP in Diffusion Model"
date: '2024-05-05 00:00:00 +800'
tags: [diffusion, generative model]
math: true
media_subpath: /_image/[Blog]/
---

After the DDPM model is proposed, the diffusion model is turning a reverse process. 
This blog would introduce three types of diffusion model, which is variational preserve (VP), variational explore (VE), and sub variation preserve diffusion model (sub-VP). 

Pre-request for the reading: 
Familiar with the diffusion model

# Score-Based Generative Model

As you may already know, the diffusion process is tryly a Stocastic Differential Equation.  


$$
d\textbf{x} = f(x,t) dt + g(t)d\textbf{w}
$$

The diffusion process maps the initial distribution $p_0(x)$ to prior distribution $p_T(x)$ through the diffsion process. 
The fascinate of diffusion process is that we can construct the initial distribution by utilizing reverse diffusion process 

$$
dx = \left[f(x,t) - g^2(t)\nabla_x\log p_t(x)\right]dt + g(t)d\omega
$$

Here, the term $\nabla_x\log p_t(x)$ is the so-called score-function existed in the statistic textbook. 

A score-based model is then training on "expecting the score function" to get the reverse SDE. 
Thus, with the known prior distribtuion $p_T(x)$ (usually be Gaussian), we can get the original distribution by applying reverse diffusion process. 

Then lets jump into the question, what are the VP, VE, sub-VP in diffusion model papers? It is about how the noise being added. 

# Variational Preserve Diffusion model 

For the DDPM, the forward diffusion process can be written as 

$$
\textbf{x}_{i} = \sqrt{1-\beta_i}\textbf{x}_{i-1} + \sqrt{\beta_i}z_{i-1}
$$

Then we can take the continuous limit and get   

$$  
\begin{array}{l}
\textbf{x}_{i} \approx \left(1-\frac{1}{2}\beta_i\right)\textbf{x}_{i-1}+ \sqrt{\beta_i}z_{i-1}  \\
\rightarrow \textbf{x}_{i} - \textbf{x}_{i-1} = -\frac{1}{2}\beta_i\textbf{x}_{i-1} +  \sqrt{\beta_i}z_{i-1}  \\
\rightarrow d\textbf{x} = -\frac{1}{2}\beta(t)\textbf{x} + \sqrt{\beta_i}d\textbf{w}
\end{array}
$$

in the ddpm setting, the SDE always has a fixed variance, this is the socalled variational preserved process. 