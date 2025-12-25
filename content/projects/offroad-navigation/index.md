---
title: "Offroad Navigation"
layout: "simple"
---

---

## Overview

Self-driving cars have come a long way from basic lane keeping assist to full autonomous capability, due to the success of Tesla and Waymo. They are pretty good at driving in urban settings, but fail miserably in offroad conditions. This is due to the absence of a structured environment, which self-driving cars take advantage of in cities.

My work as an intern at [Moon Lab](https://moonlab.iiserb.ac.in/index.html) during the Summer of 2025, focuses on testing the offroad navigation capabilities of existing methods. I implemented a visuo-motor policy inspired from the [Learning by Cheating](https://arxiv.org/abs/1912.12294) paper, and deployed it in an ATV. The project is discussed in detail in the following sections.

## Data Acquisition

The dataset consists of expert trajectories collected from simulations in [CARLA](https://carla.org/). A python script is run to spawn a vehicle and attach a camera and a lidar to it. It then sets the vehicle to *autopilot* mode to let CARLA control it using its internal *traffic manager*. As the vehicle traverses the map, the script, at every frame, takes a snapshot of the view in the camera, the lidar, and other attributes of the vehicle, such as, position, velocity,, accelaration, the current control applied and the high level command. Below is a typical episode of an expert driving through the environment.


Now, there are some problems which needs to be addressed.

<!-- docker for sensors + automating scripts
carla data collection + clever tricks
cnn implementation + tried weighted loss
testing -> pid control
carla port to  mac -->

<!-- ros, docker and lots of linux
carla
pytorch
pid controls -->