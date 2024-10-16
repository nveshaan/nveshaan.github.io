---
title: "Muscle Wave Classifier"
layout: "simple"
---

---

## Overview

Electromyography (EMG) measures the electrical activity generated by skeletal muscles during contractions, typically using surface electrodes. This technique enables the identification of specific gestures by analyzing the signals, making it particularly useful for applications in human-computer interaction. By interpreting muscle activity, devices can effectively respond to user intentions, thereby enhancing assistive technologies for individuals with mobility impairments.

To achieve high-accuracy human-computer interaction while minimizing the calibration data required from new users, a neural network is trained to classify gestures from EMG signals. This approach ensures seamless integration, allowing devices to adapt quickly to individual users and provide a more intuitive and responsive experience.

The project is divided into three main parts:

* **Data Collection**
* **Model Design and Training**
* **Real Time Interface**

## Data Acquisition

The dataset for this project has been collected from 13 subjects, using an EMG sensor from UpsideDownLabs.