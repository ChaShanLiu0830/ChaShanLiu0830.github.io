---
layout: post
title: "Diffusion Schrödinger Bridge Problem"
date: '2024-04-26 00:11:26 +0800'
tags: [paper, diffusion, generative model]
math: true
---

The diffusion model has envoke large attention these few days, however, the diffusion model requires all the density be mapped to normal gaussian distribution. If we like our model to transfer one prior distribution to another, one can utilize Schrodinger Bridge to build the connection. In this article, I'll try to share what I know about the diffusion bridge problem. 

# Introduction
## Schrodinger Bridge Problem 

The goal of schrodinger bridge problem is aimed to find the sortest path on one distribution $P_{prior}$ to another $P_{final}$.   
To formulate, we aim to find the minimum of 

$$
\pi^{*} = \text{argmin} \left\{ \; \text{KL}(\pi|p) \; | \; \pi \in \mathcal{P}_{N+1} , p \in \mathcal{P}_{0} \right\}
$$

This formula told us that we wish to drive one existed distribution to another by the formula of 


## Diffusiom Model

## Schrodinger Bridge Problem meet diffusion model 
 

## Some recent article 
