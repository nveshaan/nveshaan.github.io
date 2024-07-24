---
title: "Coding a Neural Network from scratch"
date: 2024-07-21
draft: false
tags: ["machine learning"]
---

## Overview

In this post, I will implement a neural network without using popular ML frameworks like PyTorch, TensorFlow, Keras. I will use python libraries like numpy, pandas, matplotlib to create a model that classifies hand-written digits.

### Why implement a Neural Net from scratch?

There are plenty of ML frameworks that offer out-of-the-box functionality for building neural networks. However, implementing one from scratch is a valuable exercise. It helps you understand how neural networks work under the hood, which is essential for designing effective models.

{{< alert icon="circle-info">}}
For a deep dive into neural networks, check out 3Blue1Brown's [series](https://www.3blue1brown.com/topics/neural-networks). Here, I'll focus on the practical implementation.
{{< /alert >}}
