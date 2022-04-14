---
layout: post
title:  "Predicting the outcome of a US presidential election using Bayesian Experimental Design"
date:   2019-11-06
categories: posts
external_url: https://pyro.ai/examples/elections.html
external_site: pyro.ai
thumbnail: assets/usa.svg
published: Pyro tutorial
tag: post
---

In this tutorial, we explore the use of optimal experimental design techniques to create an optimal polling strategy to predict the outcome of a US presidential election.
In the [previous tutorial](http://pyro.ai/examples/working_memory.html), we explored the use of Bayesian optimal experimental design to learn the working memory capacity of a single person. Here, we apply the same concepts to study a whole country.

We set up a Bayesian model of the winner of the election *w*, as well as the outcome *y* of any poll we may plan to conduct. The experimental design is the number of people *n* to poll in each state. To set up our exploratory model, we use historical election data 1976-2012 to construct a plausible prior and the 2016 election as our test set: we imagine that we are conducting polling just before the 2016 election.
<!--more-->
