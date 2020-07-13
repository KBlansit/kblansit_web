---
title: Uncertainty Sampling for Heatmap Localization
description: Developing uncertainty sampling algorithm for heatmap localization
date: "2019-05-02T19:47:09+02:00"
jobDate: 2020
work: [Cardiac MRI, Healthcare]
techs: [python, Keras, tensorflow]
designs: [Deep Learning, cardiac MRI, Heatmap Localization, Generalization, Active Learning, Uncertainty Sampling]
thumbnail: uncertainty_sampling/uncertainity.png
---

While there have been prior examinations of using uncertainty sampling for Deep Convolutional Neural Networks, this prior work has been limited to problems like classification. However for my problem of localization, there has not been generalization of these approaches for my problem of [heatmap localization](/portfolio/cmri_proof_of_concept/ "Heatmap Localization").

In this work, I examine three different strategies for measuring Deep Convolutional Neural Network uncertainty; 1) pseudoprobability maximum based off the Deep Convolutional Neural Network activation, 2) spatial variance of rotational entropy based off augmentations of the test images, and 3) a Bayesian uncertainty metric based off applying random dropout to the network.

I demonstrate significant improvements data-efficiency for training a Deep Convolutional Neural Network based off either our pseudoprobability maximum or spatial variance of rotational entropy metrics. Using these metrics may further enhance the ability to develop Deep Convolutional Neural Network on scarce medical data.
