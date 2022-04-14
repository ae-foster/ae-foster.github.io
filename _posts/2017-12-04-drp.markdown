---
layout: post
title:  "Sampling and inference for discrete random probability measures in probabilistic programs"
date:   2017-12-04
categories: papers
external_url: http://www.approximateinference.org/2017/accepted/Bloem-ReddyEtAl2017.pdf
external_site: approximateinference.org
authors: Benjamin Bloem-Reddy, Emile Mathieu, Adam Foster, Tom Rainforth, Yee Whye Teh, Hong Ge, María Lomelí, Zoubin Ghahramani
published: NeurIPS 2017 Workshop on Advances in Approximate Bayesian Inference
tag: paper
---

We consider the problem of sampling a sequence from a discrete random probability measure (RPM) with countable support, under (probabilistic) constraints of finite memory and computation. A canonical example is sampling from the Dirichlet Process, which can be accomplished using its stick-breaking representation and lazy initialization of its atoms. We show that efficiently lazy initialization is possible if and only if a size-biased representation of the discrete RPM is used. For models constructed from such discrete RPMs, we consider the implications for generic particle-based inference methods in probabilistic programming systems. To demonstrate, we implement SMC for Normalized Inverse Gaussian Process mixture models in Turing.
<!--more-->
