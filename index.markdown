---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

![My helpful screenshot](/assets/adamfoster.jpg){:class="main-img"}
I am a senior researcher at [Microsoft Research AI4Science](https://www.microsoft.com/en-us/research/lab/microsoft-research-ai4science/) where I work on machine learning methods for chemistry with [Frank Noé](https://www.microsoft.com/en-us/research/people/franknoe). 
I want to understand how machine learning can help us to solve critical problems in the sciences and to build new, sustainable technology.
At Microsoft, I spent two years working on electronic structure: developing an *ab initio* foundation model for wavefunctions using deep Quantum Monte Carlo.
This is a new, unsupervised approach high-accuracy quantum chemistry that can tackle multireferential problems like bond breaking.
We also worked on [extracting electron densities from our QMC wavefunctions](https://arxiv.org/abs/2409.01306).
I now work on modelling [protein dynamics](https://github.com/microsoft/bioemu).

I also have a strong interest in Bayesian experimental design and active learning.
This was the main topic of my PhD in Statistical Machine Learning at the University of Oxford, supervised by [Yee Whye Teh](https://www.stats.ox.ac.uk/~teh/index.html) and [Tom Rainforth](https://www.robots.ox.ac.uk/~twgr/) in the [Computational Stats and Machine Learning Group](http://csml.stats.ox.ac.uk/) in the [Department of Statistics](https://www.stats.ox.ac.uk/).
I was awarded the prestigious [Corcoran Memorial Prize](https://www.stats.ox.ac.uk/events/2022-corcoran-memorial-prize-lecture) for my PhD thesis.
Before starting my PhD, I studied [mathematics at Cambridge](https://www.maths.cam.ac.uk/) where my Director of Studies was [Julia Gog](http://www.damtp.cam.ac.uk/person/jrg20). 


A large part of my PhD work was on Bayesian experimental design: how do we design experiments that will be most informative about the process being investigated?
One approach is to optimize the Expected Information Gain (EIG), which can be seen as a mutual information, over the space of possible designs.
In my work, I have developed [variational methods to estimate the EIG](https://arxiv.org/abs/1903.05480), [stochastic gradient methods to optimize over designs](https://arxiv.org/abs/1911.00294), and how to obtain [unbiased gradient estimators of EIG](https://arxiv.org/abs/2005.08414). 
In more recent work, we have studied [policies that can choose a sequence of designs automatically](https://arxiv.org/abs/2103.02438).
These two talks ([Corcoran Memorial Prize Lecture](https://www.stats.ox.ac.uk/events/2022-corcoran-memorial-prize-lecture) and [SIAM minisymposium](https://www.youtube.com/watch?v=zgHE5phpytw)) offers introductions to experimental design and my research in this area.

I am also keen on [open-source code](https://github.com/ae-foster): highlights include [OneQMC](https://github.com/microsoft/oneqmc), [experimental design tools in deep probabilistic programming language Pyro](https://docs.pyro.ai/en/stable/contrib.oed.html), [forward Laplacians](https://github.com/microsoft/folx), [Redis<->Python interfacing](https://github.com/ae-foster/rdbgenerate), [reproducing SimCLR](https://github.com/ae-foster/pytorch-simclr).

<!-- To use Bayesian experimental design in practice, we have developed [a range of tools in deep probabilistic programming language Pyro](https://docs.pyro.ai/en/stable/contrib.oed.html): our aim is to allow automatic experimental design for any Pyro model. -->

<!-- Since EIG is a mutual information, I am also interested in the intersection between information theory and machine learning.
This led me to study contrastive representation learning and [the role of invariance in these methods](https://arxiv.org/abs/2010.09515), as well as [reproducing SimCLR in PyTorch](https://github.com/ae-foster/pytorch-simclr). -->

[comment]: <> (Within the OxCSML group, I have been fortunate enough to be introduced to a wide range of new ideas in machine learning. We run a reading group on the role of symmetry and equivariance in deep learning. We have also read the latest research in reinforcement learning, deep generative models and metalearning, among other topics.)

[comment]: <> (I am currently searching for jobs!)

[comment]: <> (I am on the look out for opportunities to use machine learning, make a contribution, learn, create, and share what I already know.)

[comment]: <> (Having spent some time working on Bayesian Experimental Design, I know there are a number of exciting directions that research in the field could go&mdash;natural language and language models, the connection to reinforcement learning, implicit models, and improving our theoretical understanding&mdash;as well as potential applications in science, education, politics and biotech to name a few. )

[comment]: <> (I am also keen to use deep learning and probabilistic modelling more broadly to tackle interesting problems. )
