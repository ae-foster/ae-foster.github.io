---
layout: post
title:  "Efficient Real-world Testing of Causal Decision Making via Bayesian Experimental Design for Contextual Optimisation"
date:   2022-07-13
categories: papers
external_url: https://arxiv.org/abs/2207.05250
external_site: arxiv.org
authors: Desi R. Ivanova, Joel Jennings, Cheng Zhang, Adam Foster
published: ICML 2022 Workshop on Adaptive Experimental Design and Active Learning in the Real World
thumbnail: assets/bed-context-optim-thumbnail.PNG
tag: paper
---

The real-world testing of decisions made using causal machine learning models is an essential prerequisite for their successful application. We focus on evaluating and improving contextual treatment assignment decisions: these are personalised treatments applied to e.g. customers, each with their own contextual information, with the aim of maximising a reward. In this paper we introduce a model-agnostic framework for gathering data to evaluate and improve contextual decision making through Bayesian Experimental Design. Specifically, our method is used for the data-efficient evaluation of the regret of past treatment assignments. Unlike approaches such as A/B testing, our method avoids assigning treatments that are known to be highly sub-optimal, whilst engaging in some exploration to gather pertinent information. We achieve this by introducing an information-based design objective, which we optimise end-to-end. Our method applies to discrete and continuous treatments. Comparing our information-theoretic approach to baselines in several simulation studies demonstrates the superior performance of our proposed approach.
<!--more-->
