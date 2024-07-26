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

The model uses **batch gradient descent** algorithm to find minima of cost function.

## Implementation

### Setup

As mentioned above, I will import the required Python libraries.

```py
import numpy as np
import pandas as pd
import matplotlib as plt
```

Then, I will split the [data](https://drive.google.com/file/d/1UcrPb8EZ6WFqm3xZV9saaWZnA6-doQPU/view?usp=sharing) into `train` and `test` sets, taking `m = 30000` as number of traning examples.

```py
data = pd.read_csv('/data.csv')
train = data[:30000]
test = data[30000:]

X = train.drop(columns=['label']).transpose() # input
Y = train['label'] # output

X_t = test.drop(columns=['label']).transpose() # input
Y_t = test['label'] # output
```

Now, I will **one-hot encode** the labels

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

At first, these parameters are just random numbers. When used, they produce garbage results. As the model learns to predict the correct values, it tunes them to reasonable numbers. Which, when used, will yield good results.

<!-- And lastly, the hyperparameters.

```py
epoch = 500  # no. of iterations
alpha = 0.8  # learning rate
m = 30000    # no. of training examples
``` -->

### Training

Before going into the training process, I will discuss each part separately.

#### Forward Prop

I will feed the training examples to the input layers, multiplying them by the weights and adding the bias values. This output is then input into the first hidden layer, and this process continues to the final output layer.

```py
Z1 = np.dot(W1, X[i]) + B1
A1 = 1/(1+np.exp(-Z1+1e-5)) # sigmoid

Z2 = np.dot(W2, A1) + B2
A2 = 1/(1+np.exp(-Z2+1e-5)) # sigmoid

Z3 = np.dot(W3, A2) + B3
y = np.exp(Z3+1e-5)
y /= sum(y) # softmax

# I added a small value of 10^-5 to prevent the exponent function to vanish
```

#### Back Prop

Using the chain rule of calculus, I will calculate respective partial derivatives of each node in the layers. But, thanks to `numpy`, it is a lot easier to implement the backprop.

```py
dz3 = y - Y_one.T[i]
dw3 += np.outer(dz3, A2)
db3 += dz3

da2 = np.dot(W3.T, dz3)
dz2 = A2*(1 - A2)*da2
dw2 += np.outer(dz2, A1)
db2 += dz2

da1 = np.dot(W2.T, dz2)
dz1 = A1*(1 - A1)*da1
dw1 += np.outer(dz1, X[i])
db1 += dz1
```

I'm adding up all the increments and decrements to the weights and biases, for all the training examples and update them at the end of epoch.

#### Update Params

After iterating through all the training examples, I'll update the parameters.

```py
W3 -= alpha*dw3/m
B3 -= alpha*db3/m
W2 -= alpha*dw2/m
B2 -= alpha*db2/m
W1 -= alpha*dw1/m
B1 -= alpha*db1/m
```

This process will repeat for several iterations (or epochs) to reach the minima.

### Putting it all together

```py
for run in range(epoch):

    dw1 = np.zeros((16, 784))
    db1 = np.zeros(16)
    dw2 = np.zeros((16, 16))
    db2 = np.zeros(16)
    dw3 = np.zeros((10, 16))
    db3 = np.zeros(10)

    for i in range(m):
        # Forward prop
        Z1 = np.dot(W1, X[i]) + B1
        A1 = 1/(1+np.exp(-Z1+1e-5))

        Z2 = np.dot(W2, A1) + B2
        A2 = 1/(1+np.exp(-Z2+1e-5))

        Z3 = np.dot(W3, A2) + B3
        y = np.exp(Z3+1e-5)
        y /= sum(y)

        # Back prop
        dz3 = y - Y_one.T[i]
        dw3 += np.outer(dz3, A2)
        db3 += dz3
        da2 = np.dot(W3.T, dz3)
        dz2 = A2*(1 - A2)*da2
        dw2 += np.outer(dz2, A1)
        db2 += dz2
        da1 = np.dot(W2.T, dz2)
        dz1 = A1*(1 - A1)*da1
        dw1 += np.outer(dz1, X[i])
        db1 += dz1

    # Update params
    W3 -= alpha*dw3/m
    B3 -= alpha*db3/m
    W2 -= alpha*dw2/m
    B2 -= alpha*db2/m
    W1 -= alpha*dw1/m
    B1 -= alpha*db1/m
```

## Results
