---
layout: post
title:  "A machine learning introduction to Orbformer"
date:   2025-06-25
categories: posts
authors: Adam Foster
published: Blog post
thumbnail: assets/envelope-one.png
tag: post
---

We have just released [Orbformer](https://arxiv.org/abs/2506.19960), a foundation model of molecular wavefunctions!
This is a project I have been working on over the last two-and-a-bit years as part of an amazing team.
This blog post tells the story of Orbformer from a machine learning perspective, filling in some quantum chemistry details along the way. 
<!--more-->

Let me stress at the outset that this is a personal post, not an official communication of my employer.

Natural science is arguably the most interesting area for research in machine learning and AI today.
So many of the things that we are dreaming that AI can help us with—be it new energy sources, energy storage, greener industrial processes, new drugs and vaccines, etc—require AI to contribute to a major breakthrough in natural science.
Where can AI help? At the end of the day, a scientific AI model has to accurately predict the outcomes of certain scientific experiments, otherwise it is not useful. 
If it enables computational surrogates (*in silico*) for real experiments that are cheap and sufficiently reliable, a given model has the chance to rapidly accelerate scientific progress.


Any well-trained machine learning practitioner, when entering a new domain of application, would ask what the available data sources are.
In natural science, experimental observation is the ultimate source of data.
But we simply don't have Internet-scale freely accessible experimental data about many key scientific phenomena.
The importance of this hurdle cannot be overstated, and data availability goes a long way to explain what in science has and has not seen successful application of machine learning so far.

To compensate the paucity of data, however, we have extremely well developed scientific theories in many domains.
Theory can be brought to bear in many ways: to add constraints and symmetries to models, to build Bayesian priors which guide data collection, to create simulated data as a substitute for experimental data, or to create unsupervised learning algorithms.
Whether and how to incorporate theory into scientific machine learning models remains, in my opinion, a fascinating open question in a variety of specific domains.
In this post, we focus on an unsupervised learning approach in which the training signal for the network stems solely from the equations of physics.

## The Schrödinger Equation and the Born—Oppenheimer Approximation
To get specific, we focus on the quantum mechanical properties of molecules. 
Our current understanding is that chemical properties of molecules and materials can be extremely well-described by the [Schrödinger wave equation](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation)<sup>1</sup>. 
In practice, for molecules and materials, we use the Schrödinger equation in combination with the [Born—Oppenheimer approximation](https://en.wikipedia.org/wiki/Born%E2%80%93Oppenheimer_approximation). 
This makes a distinction between the heavy nuclei in the molecule, which we denote collectively as $$\mathbf{M}$$, and the light electrons, $$\mathbf{x}$$. We treat $$\mathbf{M}$$ as a collection of fixed point charges, rather than quantum particles, and then solve the time-independent Schrödinger equation for just the electrons<sup>2</sup>

$$

\hat{H}_\mathbf{M}|\Psi\rangle = \mathcal{E} |\Psi\rangle.

$$

In this equation, $$|\Psi\rangle$$ is the wave function of the electrons and in the basis that we will choose, it is just a function $$\Psi(\mathbf{x} \mid \mathbf{M})$$ that takes the electron positions (and spins) as input and produces a scalar. Conditioning on $$\mathbf{M}$$ is usually left implicit.
We interpret $$|\Psi(\mathbf{x} \mid \mathbf{M})|^2$$ as the joint probability density of the electron positions, given the fixed charges $$\mathbf{M}$$.
There is one additional constraint, which is that $$\Psi$$ must be antisymmetric under the exchange of any pair of electrons. This explains why determinants are ubiquitous in quantum chemistry: they are antisymmetric under the exchange of any two rows.
The Hamiltonian, $$\hat{H}_\mathbf{M}$$, is an operator given by

$$

\hat{H}_\mathbf{M} = -\frac{1}{2}\nabla^2_\mathbf{x} + V(\mathbf{x},\mathbf{M})

$$

with $$\nabla^2$$ being the Laplacian and $$V$$ being the potential energy that consists of all particle—particle Coulomb potentials (this is written out in the appendix of the paper).
Then $$\mathcal{E}$$ is an *eigenvalue* of the operator $$\hat{H}_\mathbf{M}$$. There will be solutions to this equations for multiple values of $$\mathcal{E}$$.
It can be shown<sup>3</sup> that $$\hat{H}_\mathbf{M}$$ has real eigenvalues and that there exists a minimum eigenvalue, $$\mathcal{E}_0(\mathbf{M})$$.
The eigenstate corresponding to the minimum eigenvalue is termed the ground state, and most chemical systems are to be found in their ground state most of the time.

Given any wave function $$|\Psi \rangle$$ for the 
molecule $$\mathbf{M}$$, if we expand it on a basis of eigenfunctions, we can show the following variational principle

$$

\frac{\langle \Psi | \hat{H}_\mathbf{M} \Psi \rangle}{\langle \Psi |  \Psi \rangle} \ge \mathcal{E}_0(\mathbf{M}).

$$

Thus we have turned an eigenvalue problem into an optimization problem, which, from the perspective of deep learning, seems promising.
So far, we have followed textbook theory, and reached the branching off point for a wide variety of different methods in quantum chemistry.

## The problem of strongly correlated systems

[Wavefunction methods](https://en.wikipedia.org/wiki/Computational_chemistry#Ab_initio_method) are one huge category of approaches to solving the Schrödinger equation that explicitly represent $$|\Psi\rangle$$. These are generally considered expensive but very accurate methods, as opposed to something faster but (often) less accurate which does not explicitly represent $$|\Psi\rangle$$, like [density functional theory](https://en.wikipedia.org/wiki/Density_functional_theory). 
The virtue of wavefunction methods is that they are completely unsupervised (in machine learning vocab) or fully *ab initio* (in chemistry vocab).
Their high overall computational cost is their typical downside.

Within wavefunction methods, 'single-reference' methods such as [coupled cluster](https://en.wikipedia.org/wiki/Coupled_cluster) use as a starting point an approximate wavefunction that is similar to a mean-field approximation: they assume minimal correlation between the positions of different electrons. 
If this starting approximation is good, then single-reference methods can be very, very accurate at a reasonable cost for small molecules. 
But this approximation and the single-reference methods that use it start to break down unexpectedly and dramatically.
The canonical example, that we use in our paper, is the bond dissociation, in which single-reference methods transition from working extremely well to extremely poorly as a chemical bond is stretched and broken.

A scalable method for multireferential calculations would therefore be of immense value, and is what we set out to develop using the machinery of deep QMC.

## Deep Quantum Monte Carlo

Returning to the *variational principle*, we can rewrite it as an expectation

$$

\frac{\langle \Psi | \hat{H}_\mathbf{M} \Psi \rangle}{\langle \Psi | \Psi\rangle} = \mathbb{E}_\mathbf{\mathbf{x} \sim |\Psi(\mathbf{x} \mid \mathbf{M})|^2}\left[ \frac{\hat{H}_\mathbf{M} \Psi(\mathbf{x}\mid\mathbf{M})}{\Psi(\mathbf{x} \mid \mathbf{M})} \right] \ge \mathcal{E}_0(\mathbf{M}).

$$

This expectation involves sampling from the unnormalised distribution defined by the model itself, which has been referred to as the 'self-generative' property (Hermann et al., 2023). 
To make further progress, we need to select a specific family of trial wavefunctions which we can optimize.
For molecules, the use of neural networks as a wavefunction *ansatz* began with PauliNet and FermiNet (Hermann et al., 2020; Pfau et al., 2020).
These works propose a neural network that takes $$\mathbf{x}$$ as an input and uses a determinant to get the necessary antisymmetry.
Training, although severely complicated by the presence of the Laplacian within the Hamiltonian, can proceed through the general principle of stochastic gradient minimization. 
The result was that neural networks could achieve essentially exact solutions to the Schrödinger equation for small molecules.

## Transferability

To recap a specific point about PauliNet and FermiNet—the only network input was the spatial-spin co-ordinates of the electrons, $$\mathbf{x}$$. The dependence on $$\mathbf{M}$$ was implicit with a different network and a different training run for different molecular configurations.
This approach of solving a separate optimization problem for each $$\mathbf{M}$$ has been, with very limited exceptions, the norm across all other traditional wavefunction methods as well.

More recently, however, it's become clear that with neural network wavefunction models, there is no reason why we should restrict ourselves to training fresh networks for different molecules. 
Instead, we can train a single network that takes $$\mathbf{M},\mathbf{x}$$ as inputs, and train that to minimize the following objective

$$

\mathbb{E}_{\mathbf{M}\sim p_\text{train}(\mathbf{M}), \mathbf{x} \sim |\Psi(\mathbf{x}\mid \mathbf{M})|^2} \left[ \frac{\hat{H}_\mathbf{M} \Psi(\mathbf{x} \mid \mathbf{M})}{\Psi(\mathbf{x}\mid \mathbf{M})} \right].

$$

The idea here is that the single network can exploit commonalities between the electronic wavefunctions that occur in different molecules. 
Indeed, a conceptual understanding of chemistry is built heavily on the idea that some key patterns repeat themselves in the electronic structure of different molecules (bonds, lone pairs, rings, etc), but at a computational level these concepts are set aside because, if implemented crudely, they reduce accuracy.
This is the promise of a wavefunction foundation model. A model trained directly from the variational principle across a wide range of molecules could identify and re-use patterns that are too complex for humans to identify with heuristics. 
By doing so, the overall cost of solving the Schrödinger equation is amortized, leading to a radical reduction in the per-structure cost, even if the overall training cost is large.
For those of us who believe in the long-term potential of neural networks to generalize, this appears a promising direction. 

## Orbformer

We are not the first to consider making $$\mathbf{M}$$ an explicit input to a deep QMC wavefunction (Gao & Günnemann, 2023; Scherbela et al., 2024), but we sought to push this technology much further on two main fronts.
At the foundation model end, we decided to focus on scale, particularly in the quantity and diversity of training molecules in $$p_\textrm{train}$$, and the necessary technology to facilitate that scale. 
Scaling up required a redesign of the network architecture, particularly the dependence on $$\mathbf{M}$$, as well as a faster MCMC sampling algorithm to self-generate the electron data, a penalized training objective to avoid the newly identified 'determinant collapse' issue, FlashAttention combining with recent developments in fast Laplacians, and a few other tune-ups.

We also thought carefully about appropriate validation and where a model such as our might have immediate utility.
First, we put the strongly correlated systems at the forefront, since those are the ones where current methods fare worst.
We focused on the accuracy of predicting relative energy between two states (e.g. equilibrium to transition state), rather than the total energy, since the former is more chemically relevant.
And we placed much more emphasis on the trade-off between cost and accuracy, as opposed to simply showing that deep QMC wavefunctions are super accurate.

Our conclusion: Orbformer, though by no means the end of the story, marks significant progress towards a model that generalizes the electronic structure of molecules.

If you'd like to know more and see the results, check out [the paper](https://arxiv.org/abs/2506.19960)!




## References

A. Foster, Z. Schätzle, P. B. Szabó, L. Cheng, J. Köhler, G. Cassella, N. Gao, J. Li, F. Noé & J. Hermann. "An ab initio foundation model of wavefunctions that accurately describes chemical bond breaking." arXiv preprint arXiv:2506.19960 (2025).

J. Hermann, J. Spencer, K. Choo, A. Mezzacapo, W. M. C. Foulkes, D. Pfau, G. Carleo & F. Noé. "Ab initio quantum chemistry with neural-network wavefunctions." Nat. Rev. Chem. 7, 692 (2023).

J. Hermann, Z. Schätzle & F. Noé. "Deep-neural-network solution of the electronic Schrödinger equation." Nature Chemistry 12, 891 (2020).

D. Pfau, J. S. Spencer, A. G. D. G. Matthews & W. M. C. Foulkes. "Ab initio solution of the many-electron Schrödinger equation with deep neural networks." Phys. Rev. Res. 2, 033429 (2020).

N. Gao & S. Günnemann. "Generalizing neural wave functions." In International Conference on Machine Learning (2023). 

M. Scherbela, L. Gerard & P. Grohs. "Towards a transferable fermionic neural wavefunction for molecules." Nat. Commun. 15, 120 (2024).

## Footnotes

<sup>1</sup> Although this itself is not the end of the story, as it [does not account for relativity](https://en.wikipedia.org/wiki/Dirac_equation), for example.

<sup>2</sup> If you are familiar with the time-dependent version, you can arrive at the time-independent version by substituting $$\Psi(\mathbf{x},t) = \Psi(\mathbf{x})e^{-i\mathcal{E}t}$$.

<sup>3</sup> How is this shown? One starts with the [essential self-adjoint property of the Laplacian operator](https://www-users.cse.umn.edu/~garrett/m/fun/adjointness_crit.pdf), then one establishes a bound on the overall Hamiltonian which allows application of the [Kato—Rellich Theorem](https://loss.math.gatech.edu/17SPRINGTEA/7334/NOTES/section7katorellich.pdf) and also bounds the eigenvalues from below.
