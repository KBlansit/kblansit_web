---
title: Cardiac MRI Autopilot – Proof of Concept
description:
date: "2019-05-02T19:47:09+02:00"
jobDate: 2018
work: [Cardiac MRI, Healthcare]
techs: [python, Keras, tensorflow]
designs: [Deep Learning, Cardiac MRI, Heatmap Localization]
thumbnail: cmri_proof_of_concept/proof_of_concept.png
projectUrl: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6884027/
---

Cardiac MRI is an advanced imaging technique to understand heart health. Despite the medical utility of this technology, it's accessibility is often limited due to the complex anatomy of the heart, and difficulty in training technologists to acquire high quality images.

In this work, I therefore developed a deep learning (AI) based approach to automate these scans. Using several hundred scans, I developed 2D and 3D deep learning models in Keras (with tensorflow) to localize the cardiac landmarks that define the standard cardiac views. To do so, I used a technique called heatmap regression, which attempts to map an input MRI image to a gaussian "heatmap" centered at each landmark. I then validated my models in a separate test set, showing I was able to localize all the landmarks within <1cm.

My approach and resulting findings have since published in Radiology A.I.
