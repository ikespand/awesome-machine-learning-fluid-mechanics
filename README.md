# Awesome Machine Learning for Fluid Mechanics
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics) [![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/pulse) ![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/pulls)

A curated list of machine learning papers, codes, libraries, and databases applied to fluid mechanics. 

**Table of Contents**
  * [Frameworks](#frameworks)
  * [Research articles](#research-articles)
    + [Review papers](#review-papers)
    + [Physics-informed ML](#physics-informed-ml)
    + [Reduced-order modeling aided ML](#reduced-order-modeling-aided-ml)
    + [Pattern identification and experimental applications](#pattern-identification-and-experimental-applications)
    + [Others](#others)
  * [Available datasets](#available-datasets)
  * [Online video resources](#online-video-resources)
  * [Blogs, discussions and news articles](#blogs--discussions-and-news-articles)
  * [Opensource code and examples](#opensource-code-and-examples)
  * [Tutorials](#tutorials)
  * [Companies focusing on ML](#companies-focusing-on-ml)
  * [Opensource CFD codes](#opensource-cfd-codes)

## Frameworks
- [TensorFlow](https://github.com/tensorflow/tensorflow/ "TensorFlow") is a well-known machine learning library developed by Google.

- [PyTorch](https://github.com/pytorch/pytorch "PyTorch") is another framework for machine learning developed at Facebook.

- [Scikit-learn ](https://github.com/scikit-learn/scikit-learn "Scikit-learn ")is all-purpose machine learning library. It also provides the implementation of several other data analysis algorithm.

- [easyesn](https://github.com/kalekiu/easyesn "easyesn") is a very good implementation of echo state network (reservoir computing). ESN often finds its application in dynamical systems.

- [EchoTorch](https://github.com/nschaetti/EchoTorch) is another good implementation for ESN based upon PyTorch.

- [PySINDy](https://github.com/dynamicslab/pysindy "PySINDy") is a package with several implementations for the Sparse Identification of Nonlinear Dynamical systems (SINDy). It is also well suited for a dynamical system. 

- [PYPARSVD](https://github.com/Romit-Maulik/PyParSVD "PYPARSVD") is an implementation for singular value decomposition (SVD) which is distributed and parallelized which makes it efficient for large data.

## Research articles

### Review papers
1. Application of machine learning algorithms to flow modeling and optimization, 1999. ([Paper](https://web.stanford.edu/group/ctr/ResBriefs99/petros.pdf))

2. Turbulence modeling in the age of data, 2019. ([Paper](https://arxiv.org/abs/1804.00183 "Paper"))

3. A perspective on machine learning in turbulent flows, 2020. ([Paper](https://www.tandfonline.com/doi/abs/10.1080/14685248.2020.1757685 "Paper"))

4. Machine learning for fluid mechanics, 2020. ([Paper](https://www.annualreviews.org/doi/abs/10.1146/annurev-fluid-010719-060214 "Paper"))

5. A Perspective on machine learning methods in turbulence modelling, 2020. ([Paper](https://arxiv.org/abs/2010.12226 "Paper"))

6. Machine learning accelerated computational fluid dynamics, 2021. ([Paper](https://arxiv.org/abs/2102.01010 "Paper")) 


### Physics-informed ML
1. Reynolds averaged turbulence modeling using deep neural networks with embedded invariance, 2016. ([Paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/reynolds-averaged-turbulence-modelling-using-deep-neural-networks-with-embedded-invariance/0B280EEE89C74A7BF651C422F8FBD1EB "Paper"))

2. From deep to physics-informed learning of turbulence: Diagnostics, 2018. ([Paper](https://arxiv.org/abs/1810.07785 "Paper"))

3. Subgrid modelling for two-dimensional turbulence using neural networks, 2018. ([Paper](https://arxiv.org/abs/1808.02983 "Paper") | [Code](https://github.com/Romit-Maulik/ML_2D_Turbulencehttp:// "Code"))

4. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, 2019. ([Paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125 "Paper"))

5. Neural network models for the anisotropic Reynolds stress tensor in turbulent channel flow, 2019. ([Paper](https://arxiv.org/abs/1909.03591 "Paper"))

6. Data-driven fractional subgrid-scale modeling for scalar turbulence: A nonlocal LES approach, 2020. ([Paper](https://arxiv.org/abs/2012.14027 "Paper"))

7. A machine learning framework for LES closure terms, 2020. ([Paper](https://arxiv.org/abs/2010.03030 "Paper"))

8. A neural network based shock detection and localization approach for discontinuous Galerkin methods, 2020. ([Paper](https://arxiv.org/pdf/2001.08201.pdf "Paper"))

9. Stable a posteriori LES of 2D turbulence using convolutional neural networks: Backscattering analysis and generalization to higher Re via transfer learning, 2021. ([Paper](https://arxiv.org/abs/2102.11400 "Paper"))

10. Data-driven algebraic models of the turbulent Prandtl number for buoyancy-affected flow near a vertical surface, 2021. ([Paper](https://arxiv.org/abs/2104.01842))

### Reduced-order modeling aided ML
1. Reservoir computing model of two-dimensional turbulent convection, 2020. ([Paper](https://arxiv.org/abs/2001.10280))

2. Predictions of turbulent shear flows using deep neural networks, 2019. ([Paper](https://arxiv.org/abs/1905.03634 "Paper") | [Code](https://github.com/KTH-Nek5000/DeepTurbulence "Code"))

3. Reduced-order modeling of advection-dominated systems with recurrent neural networks and convolutional autoencoders, 2020. ([Paper](https://github.com/Romit-Maulik/CAE_LSTM_ROMS "Paper") | [Code](https://arxiv.org/pdf/2002.00470.pdf "Code")) 

4. Time-series learning of latent-space dynamics for reduced-order model closure, 2020. ([Paper](https://linkinghub.elsevier.com/retrieve/pii/S0167278919305536 "Paper") | [Code](https://github.com/Romit-Maulik/ML_ROM_Closures "Code"))

5. A deep learning enabler for nonintrusive reduced order modeling of fluid flows, 2019. ([Paper](https://arxiv.org/abs/1907.04945 "Paper"))

6. Echo state network for two-dimensional turbulent moist Rayleigh-Bénard convection, 2020. ([Paper](https://arxiv.org/abs/2101.11325 "Paper"))

7. DeepCFD: Efficient steady-state laminar flow approximation with deep convolutional neural networks, 2020. ([Paper](https://arxiv.org/pdf/2004.08826.pdf "Paper") | [Code](https://github.com/mdribeiro/DeepCFD "Code"))

8. From coarse wall measurements to turbulent velocity fields with deep learning, 2021. ([Paper](https://arxiv.org/abs/2103.07387))

9. Convolutional neural network and long short-term memory based reduced order surrogate for minimal turbulent channel flow, 2021. ([Paper](https://arxiv.org/abs/2010.13351), | Data: Contact authors)

### Pattern identification and experimental applications

1. Deep learning in turbulent convection networks, 2019. ([Paper](https://www.pnas.org/content/116/18/8667 "Paper"))

2. Time-resolved turbulent velocity field reconstruction using a long short-term memory (LSTM)-based artificial intelligence framework, 2019. ([Paper](https://aip.scitation.org/doi/10.1063/1.5111558 "Paper"))

3. Unsupervised deep learning for super-resolution reconstruction of turbulence, 2020. ([Paper](https://arxiv.org/abs/2007.15324 "Paper"))

4. Nonlinear mode decomposition with convolutional neural networks for fluid dynamics, 2020. ([Paper](https://arxiv.org/abs/1906.04029 "Paper"))

### Others
1. Forecasting of spatiotemporal chaotic dynamics with recurrent neural networks: a comparative study of reservoir computing and backpropagation algorithms, 2019. ([Paper](https://arxiv.org/abs/1910.05266 "Paper"))

2. Data-assisted reduced-order modeling of extreme events in complex dynamical systems, 2018. ([Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0197704 "Paper"))

3. Nonlinear mode decomposition with convolutional neural networks for fluid dynamics, 2020. ([Paper](https://arxiv.org/abs/1906.04029 "Paper"))

4. Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations, 2020. ([Paper](https://science.sciencemag.org/content/367/6481/1026 "Paper"))

## Available datasets
1. KTH FLOW: A rich dataset of different turbulent flow generated by DNS,  LES and experiments. ([Simulation data](https://www.flow.kth.se/flow-database/simulation-data-1.791810http:// "Data") | [Experimental data](https://www.flow.kth.se/flow-database/experimental-data-1.791818 "Experimental data") | [Paper-1](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/history-effects-and-near-equilibrium-in-adversepressuregradient-turbulent-boundary-layers/39C38082C380F396D004B65F438C296A "Paper-1"))

2. Vreman Research: Turbulent channel flow dataset generated from simulation, could be useful in closure modeling. ([Data](http://www.vremanresearch.nl/channel.html "Data") | [Paper-1](http://www.vremanresearch.nl/Vreman_Kuerten_Chan180_PF2014.pdf "Paper-1") | [Paper-2](http://www.vremanresearch.nl/Vreman_Kuerten_Chan590_PF2014.pdf "Paper-2"))

3. Johns Hopkins Turbulence Databases: High quality datasets for different flow problems. ([Database](http://turbulence.pha.jhu.edu/webquery/query.aspx "Database") | [Paper](https://www.tandfonline.com/doi/abs/10.1080/14685248.2015.1088656?journalCode=tjot20 "Paper"))

4. CTR Stanford: Dataset for turbulent pipe flow and boundary layer generated with DNS. ([Database](https://ctr.stanford.edu/research-data "Database") | [Paper](https://www.pnas.org/content/114/27/E5292 "Paper"))

5. sCO2: Spatial data along the tube for heated and cooled pipe under supercritical pressure. It includes around 50 cases, which is a good start for regression based model to replace correlations. ([Data](https://www.ike.uni-stuttgart.de/forschung/Ueberkritisches-CO2/dns/ "Data") | [Paper-1](https://www.sciencedirect.com/science/article/abs/pii/S0017931017353176 "Paper-1") | [Paper-2](https://www.sciencedirect.com/science/article/abs/pii/S0017931017307998 "Paper-2"))

## Online video resources
- A first course on machine learning from Nando di Freitas: Little old, recorded in 2013 but very concise and clear. ([YouTube](https://www.youtube.com/playlist?list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6 "YouTube") | [Slides](https://www.cs.ubc.ca/~nando/540-2013/lectures.html "Slides"))

- Steve Brunton has a wonderful channel for a variety of topics ranging from data analysis to machine learning applied to fluid mechanics. ([YouTube](https://www.youtube.com/c/Eigensteve/playlists "YouTube"))

- Nathan Kutz has a super nice channel devoted to applied mathematics for fluid mechanics. ([YouTube](https://www.youtube.com/channel/UCoUOaSVYkTV6W4uLvxvgiFA/videos "YouTube"))

- For beginners, a good resource to learn OpenFOAM from József Nagy. OpenFOAM can be adapted for applying ML model coupled with N-S equations (e.g. RANS/LES closure). ([YouTube](https://www.youtube.com/c/J%C3%B3zsefNagyOpenFOAMGuru/playlists "YouTube"))

## Blogs, discussions and news articles
1. [Convolutional Neural Networks for Steady Flow Approximation](https://www.autodesk.com/research/publications/convolutional-neural-networks), 2016. (Autodesk)

2. [CFD + Machine learning for super fast simulations](https://www.reddit.com/r/CFD/comments/5n91uz/cfd_machine_learning_for_super_fast_simulations/ "CFD + Machine learning for super fast simulations"), 2017. (Reddit)

3. [What is the role of Artificial Intelligence (AI) or Machine Learning in CFD?](https://www.quora.com/What-is-the-role-of-Artificial-Intelligence-AI-or-Machine-Learning-in-CFD "What is the role of Artificial Intelligence (AI) or Machine Learning in CFD?"), 2017. (Quora)

4. [Supercomputing simulations and machine learning help improve power plant](https://www.eurekalert.org/pub_releases/2018-08/gcfs-ssa082018.php "Supercomputing simulations and machine learning help improve power plant"), 2018.

5. [When CAE Meets AI: Deep Learning For CFD Simulations](https://blog.theubercloud.com/when-cae-meets-ai-deep-learning-for-cfd-simulations), 2019. (Ubercloud)

6. [Machine Learning in Computational Fluid Dynamics](https://towardsdatascience.com/machine-learning-in-computational-fluid-dynamics-7018941414b9), 2020. (TowardsDataScience)

7. [Studying the nature of turbulence with Neural Concept's deep learning platform](https://www.numeca.com/readnews/article/616 "Studying the nature of turbulence with Neural Concept's deep learning platform"), 2020. (Numeca)

8. [A case for machine learning in CFD](https://tinyurl.com/2f6u8jab), 2020. (Medium)

9. [Machine Learning for Accelerated Aero-Thermal Design in the Age of Electromobility](https://blog.engys.com/machine-learning-for-accelerated-aero-thermal-design-in-the-age-of-electromobility/ "Machine Learning for Accelerated Aero-Thermal Design in the Age of Electromobility"), 2020. (Engys)

10. [A general purpose list for tranisitioning to data science and ML](https://ikespand.github.io/posts/resources_ml/), 2021. 

11. [A compiled list of projects from NVIDIA where AI and CFD were used](https://developer.nvidia.com/taxonomy/term/727), 2021.

## Opensource code and examples
- Repositiory [machine-learning-applied-to-cfd](https://github.com/AndreWeiner/machine-learning-applied-to-cfd "machine-learning-applied-to-cfd") has some excellent examples to begin with CFD and ML.

- Repository [Computational-Fluid-Dynamics-Machine-Learning-Examples](https://github.com/loliverhennigh/Computational-Fluid-Dynamics-Machine-Learning-Examples "Computational-Fluid-Dynamics-Machine-Learning-Examples") has an example implementation for predicting drag from the boundary conditions alongside predicting the velocity and pressure field from the boundary conditions.

- [Image Based CFD Using Deep Learning](https://github.com/IllusoryTime/Image-Based-CFD-Using-Deep-Learning "Image Based CFD Using Deep Learning")

## Tutorials
-  [Tutorial on the **Proper Orthogonal Decomposition (POD)** by Julien Weiss](https://depositonce.tu-berlin.de/bitstream/11303/9456/5/podnotes_aiaa2019.pdf "Tutorial on the **Proper Orthogonal Decomposition (POD)** by Julien Weiss"): A step by step tutorial including the data and a Matlab implementation. POD is often used for dimensionality reduction.

## Companies focusing on ML
- [Neural Concepts](https://neuralconcept.com/ "Neural Concepts") is harnessing deep learning for the accelerated simulation and design.
- [Flowfusic](https://www.flowfusic.com/about) is a cloud based provider for CFD simulation based upon OpenFOAM. They are exploring some use cases for AI and CFD.

## Opensource CFD codes
Following opensource CFD codes can be adapated for synthetic data generation. Some of them can also be used for RANS/LES closure modeling based upon ML.
1. [Nek5000](https://nek5000.mcs.anl.gov/)
2. [OpenFOAM](https://www.openfoam.com/)
3. [PyFr](http://www.pyfr.org/ "PyFr")
4. [Nektar++](https://www.nektar.info/ "Nektar++")
5. [Flexi](https://www.flexi-project.org/ "Flexi")
6. [SU2](https://su2code.github.io/ "SU2")
