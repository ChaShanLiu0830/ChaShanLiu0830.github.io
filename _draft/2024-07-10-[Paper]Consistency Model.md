---
layout: post
title: "Consistency Models"
date: '2024-07-10 00:00:00 +800'
tags: [diffusion, generative model]
math: true
# media_subpath: /_image/2024-05-04-[Implement]Improving Diffusion Inverse Problem Solving
# with Decoupled Noise Annealing/
---


# Introduction

This paper address the issue of one-step or few-step diffusion model by constrain their consistency model features that every points on the trajectory meets the same end point. This paper suggest a new class of generative model as it function the probabilistic flow ODE rather than SDE. 

# Abstract

# Method 
On the PF ODE, the trajectroy is consistency says that on the trajectory the $f(x,t) = f(x,t')$ for $t\neq t'$. 
The consistency function in the paper is defined as 
$$\mathcal{F}_{\theta}(x) = c_{skip}(t)\; \textbf{x} + c_{out}(t)\;\mathcal{F}_{\theta}(x)$$

# Impact

This paper open a great 




