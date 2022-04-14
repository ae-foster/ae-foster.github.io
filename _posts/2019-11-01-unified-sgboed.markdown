---
layout: post
title:  "A Unified Stochastic Gradient Approach to Designing Bayesian-Optimal Experiments"
date:   2019-11-01
categories: papers
external_url: https://arxiv.org/abs/1911.00294
external_site: arxiv.org
authors: Adam Foster, Martin Jankowiak, Matthew O'Meara, Yee Whye Teh, Tom Rainforth
published: AISTATS 2020
thumbnail: assets/sgboed.png
tag: paper
---

We introduce a fully stochastic gradient based approach to Bayesian optimal experimental design (BOED). Our approach utilizes variational lower bounds on the expected information gain (EIG) of an experiment that can be simultaneously optimized with respect to both the variational and design parameters. This allows the design process to be carried out through a single unified stochastic gradient ascent procedure, in contrast to existing approaches that typically construct a pointwise EIG estimator, before passing this estimator to a separate optimizer. We provide a number of different variational objectives including the novel adaptive contrastive estimation (ACE) bound. Finally, we show that our gradient-based approaches are able to provide effective design optimization in substantially higher dimensional settings than existing approaches. 
<!--more-->
