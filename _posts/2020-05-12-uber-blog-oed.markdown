---
layout: post
title:  "Announcing a New Framework for Designing Optimal Experiments with Pyro"
date:   2020-05-12
categories: posts
external_url: https://eng.uber.com/oed-pyro-release/
external_site: uber.com
thumbnail: assets/oed-life-cycle.png
authors: Adam Foster, Martin Jankowiak
published: Uber Engineering Blog
tag: post
---

Experimentation is one of humanity’s principal tools for learning about our complex world. Advances in knowledge from medicine to psychology require a rigorous, iterative process in which we formulate hypotheses and test them by collecting and analyzing new evidence. At Uber, for example, experiments play an important role in the product development process, allowing us to roll out new variations that help us improve the user experience. Sometimes, we can rely on expert knowledge in designing these types of experiments, but for experiments with hundreds of design parameters, high-dimensional or noisy data, or where we need to adapt the design in real time, these insights alone won’t necessarily be up to the task. 

To this end, AI researchers at Uber are looking into new methodologies for improving experimental design. One area of research we’ve recently explored makes use of optimal experimental design (OED), an established principle based on information theory that lets us automatically select designs for complex experiments.
<!--more-->
