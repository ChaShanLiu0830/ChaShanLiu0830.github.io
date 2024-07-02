---
layout: post
title: "&#91;Paper&#93; Guiding a Diffusion Model with a Bad Version of Itself"
date: '2024-07-02 11:00:00 +0800'
tags: [diffusion, machine_learning]
---

# Introduction

Diffusion models have become an essential component in machine learning, particularly in areas like image and text generation. A recent paper introduces an innovative technique to improve diffusion model performance by using a less trained version of the model itself as guidance. This novel approach, termed autoguidance, promises to address alignment issues and enhance output quality.

# Abstract

This paper aims to guide diffusion model generation by using a less trained model. The discovery reveals that classifier guidance leads the model to concentrate on high-probability regions, while classifier-free guidance (CFG) steers it towards highly related areas.

# Problem to Solve

The researchers first investigate how CFG works, showing that while CFG can boost class distribution through high-quality distributions, it tends to concentrate the model on high-probability areas. They also identified that using an unconditioned model as a guide does not align with the conditioned task, causing the unconditional diffusion model to overshoot the desired conditional distribution, resulting in a skewed distribution.

# Method

The authors propose a method called autoguidance, which uses a not-well-trained model for guidance. This method aligns the classifier target with the main model. The "not-well-trained" model can be interpreted as one with fewer features, early-stage model weights, or a model with large drop-out rates.

# Conclusion

In summary, the authors present a novel approach to enhance diffusion model generation by leveraging a less trained version of the model itself. This technique, autoguidance, demonstrates significant potential in refining model outputs and addressing distribution alignment issues.

# Some Ideas

What constitutes good guidance and how to determine the upper bounds of effective guidance.

# Reference

[Guiding a Diffusion Model with a Bad Version of Itself](https://arxiv.org/pdf/2406.02507)