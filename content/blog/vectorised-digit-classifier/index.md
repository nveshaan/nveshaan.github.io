---
title: "Vectorising the Digit Classifier"
date: 2024-07-30
draft: false
tags: ["machine learning"]
---

## Overview

In the previous post, I trained a neural network to classify handwritten digits, achieving an accuracy of approximately 85% on the test data. The training process took more than an hour. To expedite the training process, I will now vectorize the implementation.

## Implementation

Instead of iterating through the training examples one by one, I will stack them up in a matrix X and one-hot encode the labels in Y. Then I can multiply X with W and add B to get the output y.

Before vectorising, I multiplied the weights with the training examples and added the bias values in the following way

{{< katex >}}

$$
\{
    \begin{bmatrix}
    w_{11} & w_{12} & w_{13} & w_{14}\newline
    w_{21} & w_{22} & w_{23} & w_{24}\newline
    w_{31} & w_{32} & w_{33} & w_{34}
    \end{bmatrix}
    \begin{bmatrix}
    x_{11}\newline
    x_{12}\newline
    x_{13}\newline
    x_{14}
    \end{bmatrix}
    +
    \begin{bmatrix}
    b_1\newline
    b_2\newline
    b_3
    \end{bmatrix}
\}
$$

$$
\{
    =
    \begin{bmatrix}
    w_{11}x_{11}+w_{12}x_{12}+w_{13}x_{13}+w_{14}x_{14}+b_1\newline
    w_{21}x_{11}+w_{22}x_{12}+w_{23}x_{13}+w_{24}x_{14}+b_2\newline
    w_{31}x_{11}+w_{32}x_{12}+w_{33}x_{13}+w_{34}x_{14}+b_3
    \end{bmatrix}
    =
    \begin{bmatrix}
    a_1\newline
    a_2\newline
    a_3
    \end{bmatrix}
    \}
$$

Now, I can multiply all the training examples with the weights and add the bias values in one step

$$
\{
    \begin{bmatrix}
    w_{11} & w_{12} & w_{13} & w_{14}\newline
    w_{21} & w_{22} & w_{23} & w_{24}\newline
    w_{31} & w_{32} & w_{33} & w_{34}
    \end{bmatrix}
    \begin{bmatrix}
    x_{11} & x_{21} & x_{31} & x_{41}\newline
    x_{12} & x_{22} & x_{32} & x_{42}\newline
    x_{13} & x_{23} & x_{33} & x_{43}\newline
    x_{14} & x_{24} & x_{34} & x_{44}
    \end{bmatrix}
    +
    \begin{bmatrix}
    b_1\newline
    b_2\newline
    b_3
    \end{bmatrix}
\}
$$

$$
\{
    =
    \begin{bmatrix}
    a_{11} & a_{21} & a_{31} & a_{41}\newline
    a_{12} & a_{22} & a_{32} & a_{42}\newline
    a_{13} & a_{23} & a_{33} & a_{43}
    \end{bmatrix}
\}
$$

In the code, I got rid of one `for` loop.

```py
for run in range(epoch):

    # forward prop
    Z1 = np.dot(W1, X) + B1
    A1 = 1/(1+np.exp(-Z1+1e-5))

    Z2 = np.dot(W2, A1) + B2
    A2 = 1/(1+np.exp(-Z2+1e-5))

    Z3 = np.dot(W3, A2) + B3
    y = np.exp(Z3+1e-5)
    y /= sum(y)

    # back prop
    dz3 = y - Y_one
    dw3 = np.dot(dz3, A2.T)/m
    db3 = sum(dz3.T)/m
    da2 = np.matmul(W3.T, dz3)
    dz2 = A2*(1-A2)*da2
    dw2 = np.dot(dz2, A1.T)/m
    db2 = sum(dz2.T)/m
    da1 = np.matmul(W2.T, dz2)
    dz1 = A1*(1-A1)*da1
    dw1 = np.dot(dz1, X.T)/m
    db1 = sum(dz1.T)/m

    # update params
    W3 -= alpha*dw3
    B3 -= alpha*db3.reshape(-1, 1)
    W2 -= alpha*dw2
    B2 -= alpha*db2.reshape(-1, 1)
    W1 -= alpha*dw1
    B1 -= alpha*db1.reshape(-1, 1)
```

## Conclusion

The training process became 15x faster in the vectorised form, producing similar results to the non-vectorised form.
