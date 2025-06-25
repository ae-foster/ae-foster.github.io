---
layout: post
title:  "A machine learning introduction to Orbformer"
date:   2025-06-25
categories: posts
authors: Adam Foster
published: Blog post
tag: post
---

We have just released [Orbformer](arxiv.org), a foundation model of molecular wavefunctions!
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
Our current understanding is that chemical properties of molecules and materials can be extremely well-described by the [Schrödinger wave equation](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation)^1. 
In practice, for molecules and materials, we use the Schrödinger equation in combination with the [Born—Oppenheimer approximation](https://en.wikipedia.org/wiki/Born%E2%80%93Oppenheimer_approximation). 
This makes a distinction between the heavy nuclei in the molecule, which we denote collectively as $$\mathbf{M}$$, and the light electrons, $$\mathbf{x}$$. We treat $$\mathbf{M}$$ as a collection of fixed point charges, rather than quantum particles, and then solve the time-independent Schrödinger equation for just the electrons^2

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
It can be shown^3 that $$\hat{H}_\mathbf{M}$$ has real eigenvalues and that there exists a minimum eigenvalue, $$\mathcal{E}_0(\mathbf{M})$$.
The eigenstate corresponding to the minimum eigenvalue is termed the ground state, and most chemical systems are to be found in their ground state most of the time.

Given any wave function $$|\Psi\rangle$$ for the molecule $$\mathbf{M}$$, if we expand it on a basis of eigenfunctions, we can show the following *variational principle*

$$

\frac{\langle \Psi | \hat{H}_\mathbf{M} \Psi \rangle}{\langle \Psi |  \Psi \rangle} \ge \mathcal{E}_0(\mathbf{M}).

$$

Thus we have turned an eigenvalue problem into an optimization problem, which, from the perspective of deep learning, seems promising.
So far, we have followed textbook theory, and reached the branching off point for a wide variety of different methods in quantum chemistry, which we will follow up on later.

## Deep Quantum Monte Carlo

We can take the variational principle and rewrite it as an expectation

$$

\frac{\langle \Psi | \hat{H}_\mathbf{M} \Psi \rangle}{\langle \Psi | \Psi\rangle} = \mathbb{E}_\mathbf{\mathbf{x} \sim |\Psi(\mathbf{x} \mid \mathbf{M})|^2}\left[ \frac{\hat{H}_\mathbf{M} \Psi(\mathbf{x}\mid\mathbf{M})}{\Psi(\mathbf{x} \mid \mathbf{M})} \right] \ge \mathcal{E}_0(\mathbf{M}).

$$

This expectation involves sampling from the unnormalised distribution defined by the model itself, which has been referred to as the 'self-generative' property (Hermann et al., 2023). 
To make further progress, we need to select a specific family of trial wavefunctions which we can optimize.
For molecules, the use of neural networks as a wavefunction *ansatz* began with PauliNet and FermiNet (cite).
These works propose a neural network that takes $$\mathbf{x}$$ as an input and uses a determinant to get the necessary antisymmetry.
Training, although severely complicated by the presence of the Laplacian within the Hamiltonian, can proceed through the general principle of minimization. 
The result was that neural networks could achieve essentially exact solutions to the Schrödinger equation for small molecules.





## References

arXiv preprints our paper

J. Hermann, J. Spencer, K. Choo, A. Mezzacapo, W. M. C. Foulkes, D. Pfau, G. Carleo & F. Noé. "Ab initio quantum chemistry with neural-network wavefunctions." Nat. Rev. Chem. 7, 692 (2023).

J. Hermann, Z. Schätzle & F. Noé. "Deep-neural-network solution of the electronic Schrödinger equation." Nature Chemistry 12, 891 (2020).

D. Pfau, J. S. Spencer, A. G. D. G. Matthews & W. M. C. Foulkes. "Ab initio solution of the many-electron Schrödinger equation with deep neural networks." Phys. Rev. Res. 2, 033429 (2020).

## Footnotes

^1 Although this itself is not the end of the story, as it [does not account for relativity](https://en.wikipedia.org/wiki/Dirac_equation), for example.

^2 If you are familiar with the time-dependent version, you can arrive at the time-independent version by substituting $$\Psi(\mathbf{x},t) = \Psi(\mathbf{x})e^{-i\mathcal{E}t}$$.

^3 How is this shown? One starts with the essential [self-adjoint property of the Laplacian operator](https://www-users.cse.umn.edu/~garrett/m/fun/adjointness_crit.pdf), then one establishes a bound on the overall Hamiltonian which allows application of the [Kato—Rellich Theorem](https://loss.math.gatech.edu/17SPRINGTEA/7334/NOTES/section7katorellich.pdf) and also bounds the eigenvalues from below.
