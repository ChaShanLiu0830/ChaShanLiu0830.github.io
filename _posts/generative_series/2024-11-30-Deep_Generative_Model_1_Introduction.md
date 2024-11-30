---
layout: post
title: "[Series]Deep Genertive Model Series: 1.Introduction"
date: '2024-11-30 00:00:00 +800'
tags: [diffusion, generative model]
categories: generative_series
permalink: /generative_series/introduction/
math: true
# media_subpath: /_image/2024-05-04-[Implement]Improving Diffusion Inverse Problem Solving
# with Decoupled Noise Annealing/
---

# 1. Introduction

Generative modeling aims to understand and replicate the underlying distribution of data, enabling the creation of new, realistic samples. Traditional approaches like Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) have significantly advanced this field. However, diffusion-based models, particularly Denoising Diffusion Probabilistic Models (DDPM) and their continuous-time extensions through Stochastic Differential Equations (SDEs), have recently emerged as powerful alternatives, demonstrating superior performance across diverse modalities such as images, audio, and text.

The journey began with DDPM, which introduced a discrete-time diffusion process that methodically adds noise to data and learns to reverse this process to generate samples. Building upon this foundation, Song et al. extended the framework to continuous-time SDEs, offering a more flexible and theoretically robust approach. This evolution naturally led to the development of Probability Flow Ordinary Differential Equations (PF-ODE), a deterministic counterpart to the inherently stochastic reverse diffusion process. The introduction of PF-ODE set the stage for Consistency Models, which seek to accelerate the sampling process by ensuring consistency across different time steps, thereby reducing computational overhead without compromising generative quality.

This review meticulously examines the mathematical frameworks underpinning DDPM, SDE-based generative models, PF-ODE, and Consistency Models. By dissecting the roles of stochastic processes, score functions, deterministic flows, and advanced ODE solvers, we aim to provide a unified mathematical perspective that highlights the strengths and interconnections of these methodologies.


