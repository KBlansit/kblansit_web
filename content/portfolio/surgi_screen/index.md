---
title: Cardiac MRI Autopilot – Proof of Concept
description: Initial proof of concept for localizing cardiac landmarks
date: "2019-05-02T19:47:09+02:00"
jobDate: 2018
work: [surgery, healthcare]
techs: [python, Alexa, flask]
designs: [Human interaction, health interoperability, ]
thumbnail: surgi_screen/black_alexa.jpg
projectUrl: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5977698/
---

For a Robotics in Healthcare class I took from UC San Diego's CS department, our term project was to design and prototype a healthcare technology. Given my background in healthcare, I knew of the importance health systems were placing on developing post surgical monitoring programs, such as hiring nurses to monitor patients after they were discharged to their homes. However, these programs are often highly labor intensive, and therefore often difficult to implement. I therefore design and implement a framework to better enable remote monitoring system in the hopes supplementing these readmission programs.

To develop this platform, I chose the Alexa virtual assistant as framework to support this project. Using FLASK, I developed an REST API to support custom dialogue trees (defined as a series of YAML files), allowing the program to respond to different user responses. Specific behavior included patient identity verification.

Since integration of healthcare technology is often a primary limitation to its adoption, I prototyped integration of my program with the existing FHIR standard to communicate with modern electronic health record systems, such as EPIC/Cerner/Allscripts.

This project was distinguished for design and creativity by the American Medical Informatics Association.

{{< youtube 05MgJHWA-ns >}}
