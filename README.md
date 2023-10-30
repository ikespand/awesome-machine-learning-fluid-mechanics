# Awesome Machine Learning for Fluid Mechanics

 [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics) [![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/pulse) ![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/pulls) [![GitHub pull-requests](https://img.shields.io/github/issues-pr/ikespand/awesome-machine-learning-fluid-mechanics.svg)](https://GitHub.com/ikespand/awesome-machine-learning-fluid-mechanics/pull/)

[![GitHub stars](https://img.shields.io/github/stars/ikespand/awesome-machine-learning-fluid-mechanics)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/stargazers) [![GitHub forks](https://badgen.net/github/forks/ikespand/awesome-machine-learning-fluid-mechanics/)](https://GitHub.com/ikespand/awesome-machine-learning-fluid-mechanics/network/) [![GitHub watchers](https://badgen.net/github/watchers/ikespand/awesome-machine-learning-fluid-mechanics/)](https://GitHub.com/ikespand/awesome-machine-learning-fluid-mechanics/watchers/) [![GitHub contributors](https://badgen.net/github/contributors/ikespand/awesome-machine-learning-fluid-mechanics)](https://GitHub.com/ikespand/awesome-machine-learning-fluid-mechanics/graphs/contributors/) 

 [![Maintainers Wanted](https://img.shields.io/badge/maintainers-wanted-red.svg)](mailto:spandey.ike@gmail.com)

----

A curated list of machine learning papers, codes, libraries, and databases applied to fluid mechanics. This list in no way a comprehensive, therefore, if you observe something is missing then please feel free to add it here while adhereing to [contributing guidelines](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/blob/main/CONTRIBUTING.md).

**Table of Contents**
- [Awesome Machine Learning for Fluid Mechanics](#awesome-machine-learning-for-fluid-mechanics)
  * [Frameworks](#frameworks)
  * [Research articles](#research-articles)
    + [Editorials](#editorials)
    + [Review papers](#review-papers)
    + [Quantum Machine Learning](#quantum-machine-learning)
    + [Interpreted (/Explainable) Machine Learning](#interpreted-explainable-machine-learning)
    + [Physics-informed ML](#physics-informed-ml)
    + [Reduced-order modeling aided ML](#reduced-order-modeling-aided-ml)
    + [Pattern identification, Super-resolution and experimental applications](#pattern-identification--super-resolution-and-experimental-applications)
    + [Reinforcement learning](#reinforcement-learning)
    + [Geometry optimization/ generation](#geometry-optimization--generation)
    + [Others](#others)
  * [ML-focused events](#ml-focused-events)
  * [Available datasets](#available-datasets)
  * [Online resources](#online-resources)
  * [Blogs, discussions and news articles](#blogs--discussions-and-news-articles)
  * [Ongoing researchs, projects and labs](#ongoing-researchs--projects-and-labs)
  * [Opensource codes, tutorials and examples](#opensource-codes--tutorials-and-examples)
  * [Companies focusing on ML](#companies-focusing-on-ml)
  * [Opensource CFD codes](#opensource-cfd-codes)
  * [Support Forums](#support-forums)
----

## Frameworks
1. [TensorFlow](https://github.com/tensorflow/tensorflow/ "TensorFlow") is a well-known machine learning library developed by Google.

2. [PyTorch](https://github.com/pytorch/pytorch "PyTorch") is another framework for machine learning developed at Facebook.

3. [Scikit-learn ](https://github.com/scikit-learn/scikit-learn "Scikit-learn ")is all-purpose machine learning library. It also provides the implementation of several other data analysis algorithm.

4. [easyesn](https://github.com/kalekiu/easyesn "easyesn") is a very good implementation of echo state network (ESN aka. reservoir computing). ESN often finds its application in dynamical systems.

5. [EchoTorch](https://github.com/nschaetti/EchoTorch) is another good implementation for ESN based upon PyTorch.

6. [flowTorch](https://github.com/FlowModelingControl/flowtorch) is a Python library for analysis and reduced order modeling of fluid flows.

7. [neurodiffeq](https://github.com/NeuroDiffGym/neurodiffeq) is a Python package for solving differential equations with neural networks. 

8. [SciANN](https://github.com/sciann/sciann) is a Keras wrapper for scientific computations and physics-informed deep learning.

9. [PySINDy](https://github.com/dynamicslab/pysindy) is a package with several implementations for the Sparse Identification of Nonlinear Dynamical systems (SINDy). It is also well suited for a dynamical system. 

10. [smarties](https://github.com/cselab/smarties) is a Reinforcement Learning (RL) software designed high-performance C++ implementations of deep RL learning algorithms including V-RACER, CMA, PPO, DQN, DPG, ACER, and NAF.

11. [DRLinFluids](https://github.com/venturi123/DRLinFluids)is a flexible Python package that enables the application of Deep Reinforcement Learning (DRL) techniques to Computational Fluid Dynamics (CFD).  [[Paper-1](https://doi.org/10.1063/5.0103113), [Paper-2](https://doi.org/10.1063/5.0152777)]

12. [PyDMD](https://github.com/mathLab/PyDMD) is a python package for dynamic mode decomposition which is often used for reduced order modelling now.

13. [PYPARSVD](https://github.com/Romit-Maulik/PyParSVD "PYPARSVD") is an implementation for singular value decomposition (SVD) which is distributed and parallelized which makes it efficient for large data.

14. [turbESN](https://github.com/flohey/turbESN) is a python-based package which relies on PyTorch for ESN as a backend which supports fully autonomous and teacher forced ESN predictions.

## Research articles

### Editorials 
1. [Editorial: Machine Learning and Physical Review Fluids: An Editorial Perspective](https://journals.aps.org/prfluids/pdf/10.1103/PhysRevFluids.6.070001), 2021.

### Review papers
1. Application of machine learning algorithms to flow modeling and optimization, 1999. ([Paper](https://web.stanford.edu/group/ctr/ResBriefs99/petros.pdf))

2. Turbulence modeling in the age of data, 2019. ([arXiv](https://arxiv.org/abs/1804.00183 "Paper"))

3. A perspective on machine learning in turbulent flows, 2020. ([Paper](https://www.tandfonline.com/doi/abs/10.1080/14685248.2020.1757685 "Paper"))

4. Machine learning for fluid mechanics, 2020. ([Paper](https://www.annualreviews.org/doi/abs/10.1146/annurev-fluid-010719-060214 "Paper"))

5. A Perspective on machine learning methods in turbulence modelling, 2020. ([arXiv](https://arxiv.org/abs/2010.12226 "Paper"))

6. Machine learning accelerated computational fluid dynamics, 2021. ([arXiv](https://arxiv.org/abs/2102.01010 "Paper")) 

7. Deep learning to replace, improve, or aid CFD analysis in built environment applications: A review, 2021. ([Paper](https://www.sciencedirect.com/science/article/pii/S0360132321007137))

8. Physics-informed machine learning, 2021. ([Paper](https://www.nature.com/articles/s42254-021-00314-5))

9. Enhancing Computational Fluid Dynamics with Machine Learning, 2022.  ([arXiv](https://arxiv.org/pdf/2110.02085.pdf) | [Paper](https://www.nature.com/articles/s43588-022-00264-7)) 

10. Applying machine learning to study fluid mechanics, 2022. ([Paper](https://link.springer.com/content/pdf/10.1007/s10409-021-01143-6.pdf))

11. Improving aircraft performance using machine learning: A review, 2022. ([arXiv](https://arxiv.org/abs/2210.11481) | [Paper](https://www.sciencedirect.com/science/article/pii/S1270963823002511))

12. The transformative potential of machine learning for experiments in fluid mechanics, 2023. ([Paper](https://www.nature.com/articles/s42254-023-00622-y))

13. Super-resolution analysis via machine learning: a survey for fluid flows, 2023. ([Open Access Paper](https://link.springer.com/article/10.1007/s00162-023-00663-0))


### Quantum Machine Learning 
1. Machine learning and quantum computing for reactive turbulence modeling and simulation, 2021. ([Paper](https://www.sciencedirect.com/science/article/abs/pii/S0093641321000987))

2. Quantum reservoir computing of thermal convection flow, 2022. ([arXiv](https://arxiv.org/pdf/2204.13951.pdf))

3. Reduced-order modeling of two-dimensional turbulent Rayleigh-Bénard flow by hybrid quantum-classical reservoir computing, 2023. ([arXiv](https://arxiv.org/abs/2307.03053))

### Interpreted (/Explainable) Machine Learning 
1. Extracting Interpretable Physical Parameters from Spatiotemporal Systems using Unsupervised Learning, 2020. ([arXiv](https://arxiv.org/abs/1907.06011) | [Blog](https://peterparity.github.io/projects/pde_vae/))

2. An interpretable framework of data-driven turbulence modeling using deep neural networks, 2021. ([Paper](https://aip.scitation.org/doi/10.1063/5.0048909))

3. Interpreted machine learning in fluid dynamics: explaining relaminarisation events in wall-bounded shear flows, 2022, ([Paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/interpreted-machine-learning-in-fluid-dynamics-explaining-relaminarisation-events-in-wallbounded-shear-flows/C2CA43557475FF09B2FCEC06D99BB0FE) | [Data](https://datashare.ed.ac.uk/handle/10283/4424))

4. Explaining wall-bounded turbulence through deep learning. 2023. ([arXiv](https://arxiv.org/abs/2302.01250))

5. Multiscale Graph Neural Network Autoencoders for Interpretable Scientific Machine Learning, 2023 ([arXiv](https://arxiv.org/abs/2302.06186) | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0021999123006320))

6. Feature importance in neural networks as a means of interpretation for data-driven turbulence models, 2023. ([Open Access Paper](https://www.sciencedirect.com/science/article/pii/S0045793023002189))


### Physics-informed ML
1. Reynolds averaged turbulence modeling using deep neural networks with embedded invariance, 2016. ([Paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/reynolds-averaged-turbulence-modelling-using-deep-neural-networks-with-embedded-invariance/0B280EEE89C74A7BF651C422F8FBD1EB))

2. From deep to physics-informed learning of turbulence: Diagnostics, 2018. ([arXiv](https://arxiv.org/abs/1810.07785))

3. Subgrid modelling for two-dimensional turbulence using neural networks, 2018. ([arXiv](https://arxiv.org/abs/1808.02983) | [Code](https://github.com/Romit-Maulik/ML_2D_Turbulencehttp))

4. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, 2019. ([Paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125))

5. Neural network models for the anisotropic Reynolds stress tensor in turbulent channel flow, 2019. ([arXiv](https://arxiv.org/abs/1909.03591))

6. Data-driven fractional subgrid-scale modeling for scalar turbulence: A nonlocal LES approach, 2020. ([arXiv](https://arxiv.org/abs/2012.14027))

7. A machine learning framework for LES closure terms, 2020. ([arXiv](https://arxiv.org/abs/2010.03030))

8. A neural network based shock detection and localization approach for discontinuous Galerkin methods, 2020. ([arXiv](https://arxiv.org/pdf/2001.08201.pdf ))

9. Stable a posteriori LES of 2D turbulence using convolutional neural networks: Backscattering analysis and generalization to higher Re via transfer learning, 2021. ([arXiv](https://arxiv.org/abs/2102.11400 ))

10. Data-driven algebraic models of the turbulent Prandtl number for buoyancy-affected flow near a vertical surface, 2021. ([arXiv](https://arxiv.org/abs/2104.01842))

11. Convolutional Neural Network Models and Interpretability for the Anisotropic Reynolds Stress Tensor in Turbulent One-dimensional Flows, 2021. ([arXiv](https://arxiv.org/abs/2106.15757))

12. Physics-aware deep neural networks for surrogate modeling of turbulent natural convection,2021. ([arXiv](https://arxiv.org/abs/2103.03565))

13. Learned Turbulence Modelling with Differentiable Fluid Solvers, 2021. ([arXiv](https://arxiv.org/abs/2202.06988))

14. Physics-informed data based neural networks for two-dimensional turbulence, 2022. ([arXiv](https://arxiv.org/pdf/2203.02555.pdf) | [Paper](https://aip.scitation.org/doi/abs/10.1063/5.0090050))

15. Deep Physics Corrector: A physics enhanced deep learning architecture for solving stochastic differential equations, 2022. ([arXiv](https://arxiv.org/abs/2209.09750))

16. A Physics-informed Diffusion Model for High-fidelity Flow Field Reconstruction, 2022. ([arXiv](https://arxiv.org/abs/2211.14680))

17. A fast and accurate physics-informed neural network reduced order model with shallow masked autoencoder, 2022. ([arXiv](https://arxiv.org/abs/2009.11990) | [Paper](https://www.sciencedirect.com/science/article/pii/S0021999121007361))

18. FluxNet: a physics-informed learning-based Riemann solver for transcritical flows with non-ideal thermodynamics, 2022. ([Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4216629) | [Code](https://git.uwaterloo.ca/jc9wang/fluxnet))

19. An Improved Structured Mesh Generation Method Based on Physics-informed Neural Networks, 2022. ([arXiv](https://arxiv.org/abs/2210.09546))

20. Physics-Informed Neural Networks for Inverse Problems in Supersonic Flows, 2022. ([arXiv](https://arxiv.org/abs/2202.11821) | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0021999122004648))

21. Extending a Physics-Informed Machine Learning Network for Superresolution Studies of Rayleigh-Bénard Convection, 2023. ([arXiv](https://arxiv.org/abs/2307.02674))

22. Machine learning for RANS turbulence modeling of variable property flows, 2023. ([arXiv](https://arxiv.org/abs/2210.15384) | [Paper](https://www.sciencedirect.com/science/article/pii/S0045793023000609))

23. A probabilistic, data-driven closure model for RANS simulations with aleatoric, model uncertainty, 2023. ([arXiv](https://arxiv.org/abs/2307.02432))

### Reduced-order modeling aided ML
1. Reservoir computing model of two-dimensional turbulent convection, 2020. ([arXiv](https://arxiv.org/abs/2001.10280))

2. Predictions of turbulent shear flows using deep neural networks, 2019. ([arXiv](https://arxiv.org/abs/1905.03634 "Paper") | [Code](https://github.com/KTH-Nek5000/DeepTurbulence "Code"))

3. A deep learning enabler for nonintrusive reduced order modeling of fluid flows, 2019. ([arXiv](https://arxiv.org/abs/1907.04945))

4. Reduced-order modeling of advection-dominated systems with recurrent neural networks and convolutional autoencoders, 2020. ([arXiv](https://arxiv.org/pdf/2002.00470.pdf) | [Code](https://github.com/Romit-Maulik/CAE_LSTM_ROMS)) 

5. Time-series learning of latent-space dynamics for reduced-order model closure, 2020. ([Paper](https://linkinghub.elsevier.com/retrieve/pii/S0167278919305536) | [Code](https://github.com/Romit-Maulik/ML_ROM_Closures))

6. Echo state network for two-dimensional turbulent moist Rayleigh-Bénard convection, 2020. ([arXiv](https://arxiv.org/abs/2101.11325))

7. DeepCFD: Efficient steady-state laminar flow approximation with deep convolutional neural networks, 2020. ([arXiv](https://arxiv.org/pdf/2004.08826.pdf) | [Code](https://github.com/mdribeiro/DeepCFD))

8. From coarse wall measurements to turbulent velocity fields with deep learning, 2021. ([arXiv](https://arxiv.org/abs/2103.07387))

9. Convolutional neural network and long short-term memory based reduced order surrogate for minimal turbulent channel flow, 2021. ([arXiv](https://arxiv.org/abs/2010.13351), | Data: Contact authors)

10. Direct data-driven forecast of local turbulent heat flux in Rayleigh–Bénard convection, 2022. ([arXiv](https://arxiv.org/abs/2202.13129) | [arXiv](https://aip.scitation.org/doi/abs/10.1063/5.0087977) | Data: Contact authors)

11. Cost function for low‑dimensional manifold topology assessment ([Paper](https://www.nature.com/articles/s41598-022-18655-1) | [Data](https://tnfworkshop.org/data-archives/pilotedjet/ch4-air/) | [Code](https://github.com/kamilazdybal/cost-function-manifold-assessment))

12. Data-Driven Modeling for Transonic Aeroelastic Analysis, 2023. ([arXiv](https://arxiv.org/abs/2304.07046) | [Code, will be available](https://github.com/Nicola-Fonzi/pysu2DMD))

13. Predicting the wall-shear stress and wall pressure through convolutional neural networks, 2023. ([arXiv](https://arxiv.org/abs/2303.00706) | [Paper](https://www.sciencedirect.com/science/article/pii/S0142727X23000991))

14. Deep learning-based reduced order model for three-dimensional unsteady flow using mesh transformation and stitching, 2023. ([arXiv](https://arxiv.org/abs/2307.07323)| Data : Contact authors)

15. Reduced-order modeling of fluid flows with transformers, 2023. ([Paper](https://pubs.aip.org/aip/pof/article-abstract/35/5/057126/2891586/Reduced-order-modeling-of-fluid-flows-with))


### Pattern identification, Super-resolution and experimental applications

1. Deep learning in turbulent convection networks, 2019. ([Paper](https://www.pnas.org/content/116/18/8667))

2. Time-resolved turbulent velocity field reconstruction using a long short-term memory (LSTM)-based artificial intelligence framework, 2019. ([Paper](https://aip.scitation.org/doi/10.1063/1.5111558))

3. Unsupervised deep learning for super-resolution reconstruction of turbulence, 2020. ([arXiv](https://arxiv.org/abs/2007.15324))

4. Nonlinear mode decomposition with convolutional neural networks for fluid dynamics, 2020. ([arXiv](https://arxiv.org/abs/1906.04029))

5. A deep neural network architecture for reliable 3D position and size determination for Lagrangian particle tracking using a single camera, 2023. ([Open Access Paper](https://iopscience.iop.org/article/10.1088/1361-6501/ace070) | [Data](https://defocustracking.com/datasets/))

6. Sparse sensor reconstruction of vortex-impinged airfoil wake with machine learning, 2023. ([arXiv](https://arxiv.org/abs/2305.05147) | [Open Access Paper](https://link.springer.com/article/10.1007/s00162-023-00657-y))

### Reinforcement learning 
1. Automating Turbulence Modeling by Multi-Agent Reinforcement Learning, 2020. ([arXiv](https://arxiv.org/abs/2005.09023) | [Code](https://github.com/cselab/MARL_LES))

2. Deep reinforcement learning for turbulent drag reduction in channel flows, 2023. ([arXiv](https://arxiv.org/abs/2301.09889) | [Code](https://github.com/KTH-FlowAI/MARL-drag-reduction-in-wall-bounded-flows))

3. DRLinFluids -- An open-source python platform of coupling Deep Reinforcement Learning and OpenFOAM, 2023. ([arXiv](https://arxiv.org/abs/2205.12699) | [Paper](https://pubs.aip.org/aip/pof/article-abstract/34/8/081801/2846652/DRLinFluids-An-open-source-Python-platform-of?redirectedFrom=fulltext) | [Code](https://github.com/venturi123/DRLinFluids))


### Geometry optimization/ generation
1. Data-driven prediction of the performance of enhanced surfaces from an extensive CFD-generated parametric search space, 2023. ([Paper](https://iopscience.iop.org/article/10.1088/2632-2153/acca60) | Data: Contact authors)

### Others
1. Data-assisted reduced-order modeling of extreme events in complex dynamical systems, 2018. ([Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0197704))

2. Forecasting of spatiotemporal chaotic dynamics with recurrent neural networks: a comparative study of reservoir computing and backpropagation algorithms, 2019. ([arXiv](https://arxiv.org/abs/1910.05266))

3. Nonlinear mode decomposition with convolutional neural networks for fluid dynamics, 2020. ([arXiv](https://arxiv.org/abs/1906.04029))

4. Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations, 2020. ([Paper](https://science.sciencemag.org/content/367/6481/1026))

5. Engine Combustion System Optimization Using Computational Fluid Dynamics and Machine Learning: A Methodological Approach, 2021. ([Paper](https://asmedigitalcollection.asme.org/energyresources/article-abstract/143/2/022306/1086007/Engine-Combustion-System-Optimization-Using))

6. Physics guided machine learning using simplified theories, 2021. ([Paper](https://aip.scitation.org/doi/10.1063/5.0038929) | [Code](https://github.com/surajp92/PGML]))

7. Prospects of federated machine learning in fluid dynamics, 2022. ([Paper](https://aip.scitation.org/doi/10.1063/5.0104344))

8. Graph neural network-accelerated Lagrangian fluid simulation, 2022. ([Paper](https://www.sciencedirect.com/science/article/pii/S0097849322000206))

9. Learning Lagrangian Fluid Mechanics with E(3)-Equivariant Graph Neural Networks, 2023. ([arXiv](https://arxiv.org/abs/2305.15603) | [Code](https://github.com/tumaer/sph-hae))

10. An unsupervised machine-learning-based shock sensor for high-order supersonic flow solvers, 2023. ([arXiv](https://arxiv.org/abs/2308.00086) | [Code](https://github.com/andres-mg/2023_gmm_shock_sensor))


## ML-focused events
1. [International Workshop on Data-driven Modeling and Optimization in Fluid Mechanics](https://www.istm.kit.edu/dmofm.php), 2019, Karlsruhe, Germany.

2. [Symposium on Model-Consistent Data-driven Turbulence Modeling](http://turbgate.engin.umich.edu/symposium/index21.html), 2021, Virtual Event.

3. [Turbulence Modeling: Roadblocks, and the Potential for Machine Learning](https://turbmodels.larc.nasa.gov/turb-prs2021.html), 2022, USA. 

4. [Mini symposia: Analysis of Real World and Industry Applications: emerging frontiers in CFD computing, machine learning and beyond](https://www.wccm2022.org/minisymposia1217.html), 2022, Yokohama, Japan.

5. [IUTAM Symposium on Data-driven modeling and optimization in fluid mechanics](https://conferences.au.dk/iutam), 2022, Denmark.

6. [33rd Parallel Computational Fluid Dynamics International Conference](https://www.ercoftac.org/events/parcfd-2022/), 2022, Italy.

7. [Workshop: data-driven methods in fluid mechanics](https://fluids.leeds.ac.uk/2022/09/02/workshop-data-driven-methods-in-fluid-mechanics/), 2022, Leeds, UK.

8. [Lecture Series on Hands on Machine Learning for Fluid Dynamics 2023](https://www.vki.ac.be/index.php/events-ls/events/eventdetail/552/-/online-on-site-hands-on-machine-learning-for-fluid-dynamics-2023), 2023, von Karman Institute, Belgium.

9. [629 – Data-driven fluid mechanics](https://euromech.org/colloquia/colloquia-2023/629), 2024, Italy.

10. [Machine Learning for Fluid Mechanics: Analysis, Modeling, Control and Closures](https://www.datadrivenfluidmechanics.com/), February 2024, Belgium.

## Available datasets
1. KTH FLOW: A rich dataset of different turbulent flow generated by DNS,  LES and experiments. ([Simulation data](https://www.flow.kth.se/flow-database/simulation-data-1.791810http:// "Data") | [Experimental data](https://www.flow.kth.se/flow-database/experimental-data-1.791818 "Experimental data") | [Paper-1](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/history-effects-and-near-equilibrium-in-adversepressuregradient-turbulent-boundary-layers/39C38082C380F396D004B65F438C296A "Paper-1"))

2. Vreman Research: Turbulent channel flow dataset generated from simulation, could be useful in closure modeling. ([Data](http://www.vremanresearch.nl/channel.html "Data") | [Paper-1](http://www.vremanresearch.nl/Vreman_Kuerten_Chan180_PF2014.pdf "Paper-1") | [Paper-2](http://www.vremanresearch.nl/Vreman_Kuerten_Chan590_PF2014.pdf "Paper-2"))

3. Johns Hopkins Turbulence Databases: High quality datasets for different flow problems. ([Database](http://turbulence.pha.jhu.edu/webquery/query.aspx "Database") | [Paper](https://www.tandfonline.com/doi/abs/10.1080/14685248.2015.1088656?journalCode=tjot20 "Paper"))

4. CTR Stanford: Dataset for turbulent pipe flow and boundary layer generated with DNS. ([Database](https://ctr.stanford.edu/research-data "Database") | [Paper](https://www.pnas.org/content/114/27/E5292 "Paper"))

5. sCO2: Spatial data along the tube for heated and cooled pipe under supercritical pressure. It includes around 50 cases, which is a good start for regression based model to replace correlations. ([Data](https://www.ike.uni-stuttgart.de/forschung/Ueberkritisches-CO2/dns/ "Data") | [Paper-1](https://www.sciencedirect.com/science/article/abs/pii/S0017931017353176 "Paper-1") | [Paper-2](https://www.sciencedirect.com/science/article/abs/pii/S0017931017307998 "Paper-2"))

## Online resources
1. A first course on machine learning from Nando di Freitas: Little old, recorded in 2013 but very concise and clear. ([YouTube](https://www.youtube.com/playlist?list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6 "YouTube") | [Slides](https://www.cs.ubc.ca/~nando/540-2013/lectures.html "Slides"))

2. Steve Brunton has a wonderful channel for a variety of topics ranging from data analysis to machine learning applied to fluid mechanics. ([YouTube](https://www.youtube.com/c/Eigensteve/playlists "YouTube"))

3. Nathan Kutz has a super nice channel devoted to applied mathematics for fluid mechanics. ([YouTube](https://www.youtube.com/channel/UCoUOaSVYkTV6W4uLvxvgiFA/videos "YouTube"))

4. For beginners, a good resource to learn OpenFOAM from József Nagy. OpenFOAM can be adapted for applying ML model coupled with N-S equations (e.g. RANS/LES closure). ([YouTube](https://www.youtube.com/c/J%C3%B3zsefNagyOpenFOAMGuru/playlists "YouTube"))

5. A course on [Machine learning in computational fluid dynamics](https://github.com/AndreWeiner/ml-cfd-lecture) from TU Braunschweig.

6. Looking for coursed for TensorFlow, PyTorch, GAN etc. then have a look to [this wonderful YouTube channel](https://www.youtube.com/c/AladdinPersson/playlists)

7. Interviews with researchers, podcast revolving around fluid mechanics, machine learning and simulation [on this YouTube channel from Jousef Murad](https://www.youtube.com/c/TheEngiineer/videos)

8. Lecture series videos from [Data-Driven Fluid Mechanics: Combining First Principles and Machine Learning](https://www.datadrivenfluidmechanics.com/index.php/lectures-videos)


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

10. [A general purpose list for transitioning to data science and ML](https://ikespand.github.io/posts/resources_ml/), 2021. 

11. [A compiled list of projects from NVIDIA where AI and CFD were used](https://developer.nvidia.com/taxonomy/term/727), 2021.

12. [AI for CFD](https://becominghuman.ai/ai-for-cfd-intro-part-1-d1184936fc47), 2021. (Medium)

13. [4 Myths about AI in CFD](https://blogs.sw.siemens.com/simcenter/4-myths-about-ai-in-cfd/), 2021. (Siemens)

14. [Accelerating Product Development with Physics-Informed Neural Networks and NVIDIA Modulus](https://developer.nvidia.com/blog/accelerating-product-development-with-physics-informed-neural-networks-and-modulus/), 2021. (NVIDIA)

15. [NVIDIA, Rolls-Royce and Classiq Announce Quantum Computing Breakthrough for Computational Fluid Dynamics in Jet Engines](https://nvidianews.nvidia.com/news/nvidia-rolls-royce-and-classiq-announce-quantum-computing-breakthrough-for-computational-fluid-dynamics-in-jet-engines?ncid=so-nvsh-992286&dysig_tid=d8392875c185424f99626f91fc2aa79a#cid=hpc02_so-nvsh_en-eu), 2023. (NVIDIA)

16. [Develop Physics-Informed Machine Learning Models with Graph Neural Networks](https://developer.nvidia.com/blog/develop-physics-informed-machine-learning-models-with-graph-neural-networks/), 2023. (NVIDIA)

17. [The AI algorithm reduces design cycles/costs and time-to-market for advanced products](https://www.anl.gov/taps/article/activo-software-named-finalist-for-the-2023-rd-100-awards), 2023. (ANL)

18. [Closing the gap between High-Performance Computing (HPC) and artificial intelligence (AI)](https://developer.hpe.com/blog/closing-the-gap-between-hpc-and-ai/), 2023. (HPE)

## Ongoing researchs, projects and labs
1. [Center for Data-Driven Computational Physics](http://cddcp.sites.uofmhosting.net/), University of Michigan, USA.

2. [VinuesaLab](https://www.vinuesalab.com/), KTH, Sweden.

3. [DeepTurb](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-maschinenbau/profil/institute-und-fachgebiete/fachgebiet-stroemungsmechanik/carl-zeiss-stiftung-projekt-deepturb): Deep Learning in and of Turbulence, TU Ilmenau, Germany.

4. [Thuerey Group](https://ge.in.tum.de/research/): Numerical methods for physics simulations with deep learning, TU Munich, Germany.

5. [Focus Group Data-driven Dynamical Systems Analysis in Fluid Mechanics
](https://www.ias.tum.de/ias/research-areas/advanced-computation-and-modeling/data-driven-dynamical-systems-analysis-in-fluid-mechanics/), TU Munich, Germany.

6. [Mechanical and AI LAB (MAIL)](https://sites.google.com/view/barati?pli=1), Carnegie Mellon University, USA.

7. [Karniadakis's CRUNCH group](https://www.brown.edu/research/projects/crunch/current-research-0), Brown University, USA.

8. [MS 6: Machine Learning and Simulation Science](https://www.simtech2023.uni-stuttgart.de/program/minisymposia/ms6/), University of Stuttgart, Germany.

9. [Special Interest Groups: Machine Learning for Fluid Dynamics](https://www.ercoftac.org/special_interest_groups/54-machine-learning-for-fluid-dynamics/), Europe.

10. [Fukagata Lab](https://kflab.jp/en/index.php?21H05007), Keio University, Japan.

## Opensource codes, tutorials and examples
1. Repository [OpenFOAM Machine Learning Hackathon](https://github.com/OFDataCommittee/OFMLHackathon) have various projects originated from [Data Driven Modelling Special Interest Group](https://wiki.openfoam.com/Data_Driven_Modelling_Special_Interest_Group)

2. Repositiory [machine-learning-applied-to-cfd](https://github.com/AndreWeiner/machine-learning-applied-to-cfd "machine-learning-applied-to-cfd") has some excellent examples to begin with CFD and ML.

3. Repository [Computational-Fluid-Dynamics-Machine-Learning-Examples](https://github.com/loliverhennigh/Computational-Fluid-Dynamics-Machine-Learning-Examples "Computational-Fluid-Dynamics-Machine-Learning-Examples") has an example implementation for predicting drag from the boundary conditions alongside predicting the velocity and pressure field from the boundary conditions.

4. [Image Based CFD Using Deep Learning](https://github.com/IllusoryTime/Image-Based-CFD-Using-Deep-Learning "Image Based CFD Using Deep Learning")

5. [Deep-Flow-Prediction](https://github.com/thunil/Deep-Flow-Prediction) has the code for data generation, neural network training, and evaluation.

6. [TensorFlowFoam](https://github.com/argonne-lcf/TensorFlowFoam) with few tutorials on TensorFlow and OpenFoam.

7. [Reduced-order modeling of reacting flows using data-driven approaches](https://github.com/kamilazdybal/ROM-of-reacting-flows-Springer) have a Jupyter-Notebook example for the data driven modeling.

8.  [Tutorial on the Proper Orthogonal Decomposition (POD) by Julien Weiss](https://depositonce.tu-berlin.de/bitstream/11303/9456/5/podnotes_aiaa2019.pdf "Tutorial on the **Proper Orthogonal Decomposition (POD)** by Julien Weiss"): A step by step tutorial including the data and a Matlab implementation. POD is often used for dimensionality reduction.

## Companies focusing on ML
1. [Neural Concepts](https://neuralconcept.com/) is harnessing deep learning for the accelerated simulation and design.

2. [Flowfusic](https://www.flowfusic.com/about) is a cloud based provider for CFD simulation based upon OpenFOAM. They are exploring some use cases for AI and CFD.

3. [byteLAKE](https://www.bytelake.com/en/) offers a CFD Suite, which is a collection of AI models to [significantly accelerate the execution of CFD simulations](https://becominghuman.ai/ai-accelerated-cfd-computational-fluid-dynamics-how-does-bytelakes-cfd-suite-work-fea42fd0761e).

4. [NVIDIA](https://developer.nvidia.com/blog/modulus-v21-06-released-for-general-availability/) is leading with many product and libraries.

5. [NAVASTO](https://www.navasto.de/en/) has few products where they are combining AI with CFD.

## Opensource CFD codes
Following opensource CFD codes can be adapated for synthetic data generation. Some of them can also be used for RANS/LES closure modeling based upon ML.
1. [Nek5000](https://nek5000.mcs.anl.gov/)
2. [OpenFOAM](https://www.openfoam.com/)
3. [PyFr](http://www.pyfr.org/)
4. [Nektar++](https://www.nektar.info/)
5. [Flexi](https://www.flexi-project.org/)
6. [SU2](https://su2code.github.io/)
7. [code_saturne ](https://www.code-saturne.org/cms/web/)
8. [Dolfyn](https://www.dolfyn.net/)
9. [Neko](https://github.com/ExtremeFLOW/neko)

## Support Forums
1. [CFDOnline](https://www.cfd-online.com/Forums/tags/machine%20learning.html)
2. [StackExchange](https://scicomp.stackexchange.com/questions/tagged/machine-learning)
