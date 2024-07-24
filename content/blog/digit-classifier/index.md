---
title: "Coding a Neural Network from scratch"
date: 2024-07-21
draft: false
tags: ["machine learning"]
---

## Overview

In this post, I will implement a neural network without using popular ML frameworks like PyTorch, TensorFlow, Keras. I will use Python libraries like `numpy`, `pandas`, `matplotlib` to create a model that classifies hand-written digits.

### Why implement a Neural Net from scratch?

There are plenty of ML frameworks that offer out-of-the-box functionality for building neural networks. However, implementing one from scratch is a valuable exercise. It helps you understand how neural networks work under the hood, which is essential for designing effective models.

{{< alert icon="circle-info">}}
For a deep dive into neural networks, check out 3Blue1Brown's [series](https://www.3blue1brown.com/topics/neural-networks). Here, I'll focus on the practical implementation.
{{< /alert >}}

## Architecture

The model will consist of an input layer of 784 neurons, two hidden layers with 16 neurons each and an output layer with 10 neurons. This is a very simple configuration relative to modern standards.

Both hidden layers use **sigmoid** as the activation functions. The final layer goes through a **softmax** function.

## Implementation

### Setup

As mentioned above, I will import the required Python libraries.

```py
import numpy as np
import pandas as pd
import matplotlib as plt
```

Then I will split the [data](https://drive.google.com/file/d/1UcrPb8EZ6WFqm3xZV9saaWZnA6-doQPU/view?usp=sharing) into `train` and `test` sets.

```py
data = pd.read_csv('/data.csv')
train = data[:30000]
test = data[30000:]

X = train.drop(columns=['label']).transpose() # input
Y = train['label'] # output

X_t = test.drop(columns=['label']).transpose() # input
Y_t = test['label'] # output
```

Now I will **one-hot encode** the labels,

```py
Y_one = np.zeros((m, 10))

for i in range(m):
    Y_one[i][Y[i]] = 1

Y_one = Y_one.T
```

and initialise weights and biases.

```py
W1 = np.random.rand(16, 784)  # 16x784 matrix
B1 = np.random.rand(16)       # 16x1 matrix

W2 = np.random.rand(16, 16)   # 16x16 matrix
B2 = np.random.rand(16)       # 16x1 matrix

W3 = np.random.rand(10, 16)   # 10X16 matrix
B3 = np.random.rand(10)       # 10x1 matrix
```

At first, these parameters are just random numbers. When used, they produce garbage results. As the model learns to predict the right values, it tunes them to good numbers. Which, when used will produce good results.

And lastly, the hyperparameters.

```py
epoch = 500  # no. of iterations
alpha = 0.8  # learning rate
m = 30000    # no. of training examples
```

### Training

Before going to the training process, I'll first discuss each part of it separately.

#### Forward Prop
