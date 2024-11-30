---
title: "Bayes' Theorem in Machine Learning"
date: 2024-10-12
draft: false
tags: ["machine learning"]
---

## Overview

Bayes' Theorem is a fundamental concept in Probability Theory. It is widely used in fields such as statistics, machine learning, and data science, especially in the context of probabilistic inference and decision-making. In this post, I will explain how Bayes' Theorem is used in Machine Learning, by considering a simple example. But, before that, let's understand some terminology.

## Terminology

### Bayes' Theorem

It describes how to update the probability of a hypothesis based on new evidence. It provides a way to calculate the **posterior probability** of an event by combining the **prior probability** with the **likelihood**.

Mathematically, Bayes' Theorem is defined as:

{{< katex >}}

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

### Likelihood vs. Probability

Probability measures how likely a particular event is to occur, given a certain model or assumption. In Bayes' Theorem, \\( P(B) \\) represents the total probability of observing event \\(B\\).

Likelihood measures how likely the observed data is given a particular hypothesis or model. It's similar to probability with a subtle difference. Probability is about predicting outcomes, while likelihood is about fitting a model to data. In Bayes' Theorem, \\( P(B|A)\\) is the likelihood of observing data \\(B\\) given the hypothesis \\(A\\).

### Prior Probability

The prior probability represents your initial belief about the probability of a hypothesis before you have any new data or evidence. In Bayes' Theorem, \\( P(A) \\) is the prior probability of the hypothesis \\(A\\).

### Posterior Probability

The posterior probability is the updates probability of a hypothesis after observing new evidence. It combines the prior belief with the likelihood of the observed data. In Bayes' Theorem, \\( P(A|B) \\) is the posterior probability of the hypothesis \\(A\\) given the evidence \\(B\\).

## Implementation

One of the use-cases of Naive Bayes' Classifier is filtering out spam mails from your inbox. Let's see how it is implemented.

For a classifier, we need to define three parameters:
- Prior probability
- Likelihood
- Posterior probability

We do this by analysing the data we have. The dataset contains the words in the mail and their counts, and the number of spam mails we have.

Let's say we recieve 100 mails, 20 spam mails, and 80 normal mails.

Now, the prior probability is the probability of a mail being spam. It is calculated by dividing the number of spam mails by the total number of mails. So,

{{< katex >}}

$$P(spam) = \frac{20}{100} = 0.2$$

Let's assume that the normal mails we received have words like "dear", "friend", and "hi". And the spam mails have words like "money", "free", and "sign".

The likelihood is the probability of a word being in the mail. It is calculated by multiplying the frequency of the word by the prior probability. For sake of simplicity, let's assume that the liklihood of a word "friend" is low and the liklihood of a word "money" is high.

{{< katex >}}

$$P("friend" | spam) = 0.02$$
$$P("money" | spam) = 0.8$$
$$P("hi" | spam) = ...$$

And so on for all the words.

The posterior probability is the probability of a mail being spam given a word. It is calculated by multiplying the likelihood by the prior probability.

When we receive a new mail, we count the frequency of each word in the mail and use that to calculate the posterior probability.

For example, let's calculate the posterior probability for an email containing the words "spam", "money", and "money".

We already know the following:
- Prior probability \\( P(\text{spam}) = 0.2 \\).
- Likelihood \\( P(\text{spam} \mid \text{"spam"}) = 0.7 \\) and \\( P(\text{spam} \mid \text{"money"}) = 0.8 \\).

Now, using **Naive Bayes**, we assume the words are conditionally independent. So the likelihood of the words "spam", "money", and "money" appearing in the spam class is:

\\[
P(\text{"spam"}, \text{"money"}, \text{"money"} \mid \text{spam}) = \\]
\\[P(\text{"spam"} \mid \text{spam}) \times P(\text{"money"} \mid \text{spam}) \times P(\text{"money"} \mid \text{spam})
\\]
Then,
\\[
P(\text{"spam"}, \text{"money"}, \text{"money"} \mid \text{spam}) = 0.7 \times 0.8 \times 0.8 = 0.448
\\]

Next, we calculate the evidence \\( P(\text{"spam"}, \text{"money"}, \text{"money"}) \\). This is the total probability of seeing the words "spam", "money", and "money" in any email, whether it’s spam or not. For simplicity, let's assume the likelihoods for non-spam emails are:
- \\( P(\text{"spam"} \mid \text{not spam}) = 0.1 \\)
- \\( P(\text{"money"} \mid \text{not spam}) = 0.2 \\)

Thus, the likelihood of seeing these words in a non-spam email is:
$$
P(\text{"spam"}, \text{"money"}, \text{"money"} | \text{not spam}) = 0.1 \times 0.2 \times 0.2 = 0.004
$$

Now, calculate the evidence:
\\[
P(\text{"spam"}, \text{"money"}, \text{"money"}) = (P(\text{"spam"}, \text{"money"}, \text{"money"} \mid \text{spam}) \times P(\text{spam})) + (P(\text{"spam"}, \text{"money"}, \text{"money"} \mid \text{not spam}) \times P(\text{not spam}))
\\]
\\[
P(\text{"spam"}, \text{"money"}, \text{"money"}) = (0.448 \times 0.2) + (0.004 \times 0.8)\\]
\\[ = 0.0896 + 0.0032 = 0.0928
\\]

Finally, we compute the **posterior probability**:
\\[
P(\text{spam} \mid \text{"spam"}, \text{"money"}, \text{"money"})\\]
\\[ = \frac{P(\text{"spam"}, \text{"money"}, \text{"money"} \mid \text{spam}) \times P(\text{spam})}{P(\text{"spam"}, \text{"money"}, \text{"money"})}
\\]
Then,
\\[
P(\text{spam} \mid \text{"spam"}, \text{"money"}, \text{"money"})\\]
\\[ = \frac{0.448 \times 0.2}{0.0928} = \frac{0.0896}{0.0928} \approx 0.965
\\]

Thus, the **posterior probability** that the email is spam, given the words "spam", "money", and "money", is approximately **0.965**, or **96.5%**.

## Conclusion

In this post, we've implemented the Naive Bayes classifier for filtering spam emails and demonstrated how to calculate the posterior probability using Bayes' Theorem. With this example, you can see how the Naive Bayes classifier applies the prior probability, likelihood of words, and posterior probability to categorize new emails effectively.

While Naive Bayes assumes independence between features (in this case, words), which may not always hold in real-life datasets, it still performs well in practice and is computationally efficient. It's a powerful yet simple tool for many machine learning tasks, such as spam filtering, sentiment analysis, and text classification.

By continually updating our beliefs based on new data (words in emails), the Naive Bayes classifier helps us make informed decisions and efficiently identify spam emails with high accuracy.