# Awesome Machine Learning for Fluid Mechanics

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/topics/awesome-lists) [![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/pulse) ![Last commit](https://img.shields.io/github/last-commit/ikespand/awesome-machine-learning-fluid-mechanics)

![License](https://img.shields.io/badge/License-MIT-lightgrey.svg) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/pulls) [![GitHub pull requests](https://img.shields.io/github/issues-pr/ikespand/awesome-machine-learning-fluid-mechanics.svg)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/pulls)

[![GitHub stars](https://img.shields.io/github/stars/ikespand/awesome-machine-learning-fluid-mechanics)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/stargazers) [![GitHub forks](https://badgen.net/github/forks/ikespand/awesome-machine-learning-fluid-mechanics/)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/network/) [![GitHub watchers](https://badgen.net/github/watchers/ikespand/awesome-machine-learning-fluid-mechanics/)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/watchers/) [![GitHub contributors](https://badgen.net/github/contributors/ikespand/awesome-machine-learning-fluid-mechanics)](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/graphs/contributors/)

[![Maintainers Wanted](https://img.shields.io/badge/maintainers-wanted-red.svg)](mailto:spandey.ike@gmail.com)

---

A curated list of machine learning papers, codes, libraries, and databases applied to fluid mechanics. This list is not comprehensive; if something is missing, please feel free to add it while following the [contributing guidelines](https://github.com/ikespand/awesome-machine-learning-fluid-mechanics/blob/main/CONTRIBUTING.md).

## Table of Contents

- [Awesome Machine Learning for Fluid Mechanics](#awesome-machine-learning-for-fluid-mechanics)
  - [Frameworks](#frameworks)
  - [Research Articles](#research-articles)
    - [Editorials](#editorials)
    - [Review Papers](#review-papers)
    - [Applied Large Language Models](#applied-large-language-models)
    - [Quantum Machine Learning](#quantum-machine-learning)
    - [Interpretable and Explainable Machine Learning](#interpretable-and-explainable-machine-learning)
    - [Physics-informed ML](#physics-informed-ml)
    - [Reduced-Order Modeling Aided ML](#reduced-order-modeling-aided-ml)
    - [Transfer Learning](#transfer-learning)
    - [Generative AI](#generative-ai)
    - [Pattern Identification and Generation](#pattern-identification-and-generation)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Geometry Optimization or Generation](#geometry-optimization-or-generation)
    - [Others](#others)
    - [Books](#books)
  - [ML-Focused Events](#ml-focused-events)
  - [Available Datasets](#available-datasets)
  - [Online Resources](#online-resources)
  - [Blogs and News Articles](#blogs-and-news-articles)
  - [Ongoing Research, Projects and Labs](#ongoing-research-projects-and-labs)
  - [Open-Source Codes, Tutorials and Examples](#open-source-codes-tutorials-and-examples)
  - [Companies Focusing on ML](#companies-focusing-on-ml)
  - [Open-Source CFD Codes](#open-source-cfd-codes)
  - [Support Forums](#support-forums)
  - [Star History](#star-history)
  - [Contributors](#contributors)

---

## Frameworks

1. [TensorFlow](https://github.com/tensorflow/tensorflow/ "TensorFlow") is a well-known machine learning library developed by Google.

2. [PyTorch](https://github.com/pytorch/pytorch "PyTorch") is another framework for machine learning developed at Facebook.

3. [Scikit-learn](https://github.com/scikit-learn/scikit-learn "Scikit-learn") is an all-purpose machine learning library. It also provides implementations of several data analysis algorithms.

4. [easyesn](https://github.com/kalekiu/easyesn "easyesn") is a very good implementation of echo state network (ESN aka. reservoir computing). ESN often finds its application in dynamical systems.

5. [EchoTorch](https://github.com/nschaetti/EchoTorch) is another good implementation for ESN based upon PyTorch.

6. [flowTorch](https://github.com/FlowModelingControl/flowtorch) is a Python library for analysis and reduced order modeling of fluid flows.

7. [neurodiffeq](https://github.com/NeuroDiffGym/neurodiffeq) is a Python package for solving differential equations with neural networks.

8. [SciANN](https://github.com/sciann/sciann) is a Keras wrapper for scientific computations and physics-informed deep learning.

9. [PySINDy](https://github.com/dynamicslab/pysindy) is a package with several implementations for Sparse Identification of Nonlinear Dynamical Systems (SINDy). It is also well suited for dynamical systems.

10. [smarties](https://github.com/cselab/smarties) is Reinforcement Learning (RL) software designed for high-performance C++ implementations of deep RL algorithms, including V-RACER, CMA, PPO, DQN, DPG, ACER, and NAF.

<details>
<summary>Archive</summary>

11. [DRLinFluids](https://github.com/venturi123/DRLinFluids) is a flexible Python package that enables the application of Deep Reinforcement Learning (DRL) techniques to Computational Fluid Dynamics (CFD). ([Paper-1](https://doi.org/10.1063/5.0103113) | [Paper-2](https://doi.org/10.1063/5.0152777))

12. [PyDMD](https://github.com/mathLab/PyDMD) is a Python package for dynamic mode decomposition, which is often used for reduced-order modeling.

13. [PYPARSVD](https://github.com/Romit-Maulik/PyParSVD "PYPARSVD") is an implementation for singular value decomposition (SVD) which is distributed and parallelized which makes it efficient for large data.

14. [turbESN](https://github.com/flohey/turbESN) is a Python-based package that relies on PyTorch as an ESN backend and supports fully autonomous and teacher-forced ESN predictions.

15. [PyKoopman](https://github.com/dynamicslab/pykoopman) is a Python package for computing data-driven approximations to the Koopman operator. ([Paper](https://arxiv.org/abs/2306.12962))

16. [MODULO](https://github.com/mendezVKI/MODULO) is a modal decomposition package developed at the von Karman Institute for Fluid Dynamics (VKI). It offers a wide range of decomposition techniques, allowing users to choose the most appropriate method for their specific problem.

17. [DeepXDE](https://github.com/lululxvi/deepxde) is a library for scientific machine learning and physics-informed learning. DeepXDE includes PINN and DeepONet. It supports five tensor libraries as backends: TensorFlow 1.x (tensorflow.compat.v1 in TensorFlow 2.x), TensorFlow 2.x, PyTorch, JAX, and PaddlePaddle.

</details>

## Research Articles

### Editorials

1. [An Old-Fashioned Framework for Machine Learning in Turbulence Modeling](https://arxiv.org/abs/2308.00837), 2023. (Presented at NASA)

2. [Editorial: Machine Learning and Physical Review Fluids: An Editorial Perspective](https://journals.aps.org/prfluids/pdf/10.1103/PhysRevFluids.6.070001), 2021.

### Review Papers

1. From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning, 2024. ([arXiv](https://arxiv.org/abs/2410.13228))

2. Mixing Artificial and Natural Intelligence: From Statistical Mechanics to AI and Back to Turbulence, 2024. ([arXiv](https://arxiv.org/abs/2403.17993) | [Paper](https://iopscience.iop.org/article/10.1088/1751-8121/ad67bb))

3. Data-driven methods for flow and transport in porous media: A review, 2024. ([arXiv](https://arxiv.org/abs/2406.19939) | [Paper](https://www.sciencedirect.com/science/article/pii/S0017931024009797))

4. The transformative potential of machine learning for experiments in fluid mechanics, 2023. ([Paper](https://www.nature.com/articles/s42254-023-00622-y))

5. Super-resolution analysis via machine learning: a survey for fluid flows, 2023. ([Open Access Paper](https://link.springer.com/article/10.1007/s00162-023-00663-0))

6. Enhancing Computational Fluid Dynamics with Machine Learning, 2022. ([arXiv](https://arxiv.org/pdf/2110.02085.pdf) | [Paper](https://www.nature.com/articles/s43588-022-00264-7))

7. Applying machine learning to study fluid mechanics, 2022. ([Paper](https://link.springer.com/content/pdf/10.1007/s10409-021-01143-6.pdf))

8. Improving aircraft performance using machine learning: A review, 2022. ([arXiv](https://arxiv.org/abs/2210.11481) | [Paper](https://www.sciencedirect.com/science/article/pii/S1270963823002511))

9. Machine learning accelerated computational fluid dynamics, 2021. ([arXiv](https://arxiv.org/abs/2102.01010 "Paper"))

10. Deep learning to replace, improve, or aid CFD analysis in built environment applications: A review, 2021. ([Paper](https://www.sciencedirect.com/science/article/pii/S0360132321007137))

<details>
<summary>Archive</summary>

11. Physics-informed machine learning, 2021. ([Paper](https://www.nature.com/articles/s42254-021-00314-5))

12. A review on deep reinforcement learning for fluid mechanics, 2021. ([arXiv](https://arxiv.org/pdf/1908.04127) | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0045793021001407))

13. A perspective on machine learning in turbulent flows, 2020. ([Paper](https://www.tandfonline.com/doi/abs/10.1080/14685248.2020.1757685 "Paper"))

14. Machine learning for fluid mechanics, 2020. ([Paper](https://www.annualreviews.org/doi/abs/10.1146/annurev-fluid-010719-060214 "Paper"))

15. A Perspective on machine learning methods in turbulence modelling, 2020. ([arXiv](https://arxiv.org/abs/2010.12226 "Paper"))

16. Turbulence modeling in the age of data, 2019. ([arXiv](https://arxiv.org/abs/1804.00183 "Paper"))

17. Application of machine learning algorithms to flow modeling and optimization, 1999. ([Paper](https://web.stanford.edu/group/ctr/ResBriefs99/petros.pdf))

</details>

### Applied Large Language Models

1. Vortex state transitions in deep street canyons enabled by an automated large language model workflow, 2026. ([Paper](https://pubs.aip.org/aip/pof/article-abstract/38/5/055119/3389270/Vortex-state-transitions-in-deep-street-canyons?redirectedFrom=fulltext))

2. ALL-FEM: Agentic Large Language Models fine-tuned for finite element methods, 2026. ([Paper](https://www.sciencedirect.com/science/article/abs/pii/S0045782526002586) | [arXiv](https://arxiv.org/pdf/2603.21011))

3. Towards a Physics Foundation Model, 2025. ([arXiv](https://arxiv.org/abs/2509.13805) | [Code](https://github.com/FloWsnr/General-Physics-Transformer) | [Blog](https://flowsnr.github.io/blog/physics-foundation-model/))

4. Foam-Agent: Towards Automated Intelligent CFD Workflows, 2025. ([arXiv](https://arxiv.org/abs/2505.04997) | [Code](https://github.com/csml-rpi/Foam-Agent))

5. Fine-tuning a Large Language Model for Automating Computational Fluid Dynamics Simulations, 2025. ([arXiv](https://arxiv.org/abs/2504.09602) | [Paper](https://www.sciencedirect.com/science/article/pii/S2095034925000261) | [Code](https://github.com/YYgroup/AutoCFD) | [Data](https://huggingface.co/datasets/YYgroup/NL2FOAM))

6. AI Agents in Engineering Design: A Multi-Agent Framework for Aesthetic and Aerodynamic Car Design, 2025. ([Paper](https://arxiv.org/abs/2503.23315))

7. MetaOpenFOAM 2.0: Large Language Model Driven Chain of Thought for Automating CFD Simulation and Post-Processing, 2025. ([arXiv](https://arxiv.org/abs/2502.00498))

8. OpenFOAMGPT: a RAG-Augmented LLM Agent for OpenFOAM-Based Computational Fluid Dynamics, 2025. ([arXiv](https://arxiv.org/abs/2501.06327) | [Paper](https://pubs.aip.org/aip/pof/article-abstract/37/3/035120/3338372/OpenFOAMGPT-A-retrieval-augmented-large-language?redirectedFrom=fulltext))

9. Can foundation language models predict fluid dynamics?, 2025. ([Paper](https://www.sciencedirect.com/science/article/pii/S0952197625014290))

10. MetaOpenFOAM: an LLM-based multi-agent framework for CFD, 2024. ([arXiv](https://arxiv.org/abs/2407.21320) | [Code](https://github.com/Terry-cyx/MetaOpenFOAM) | [YouTube Presentation](https://www.youtube.com/watch?v=DiQbce4_OqI))

11. FLUID-LLM: Learning Computational Fluid Dynamics with Spatiotemporal-aware Large Language Models, 2024. ([arXiv](https://arxiv.org/abs/2406.04501))

### Quantum Machine Learning

1. Reduced-order modeling of two-dimensional turbulent Rayleigh-Bénard flow by hybrid quantum-classical reservoir computing, 2023. ([arXiv](https://arxiv.org/abs/2307.03053))

2. Quantum reservoir computing of thermal convection flow, 2022. ([arXiv](https://arxiv.org/pdf/2204.13951.pdf))

3. Machine learning and quantum computing for reactive turbulence modeling and simulation, 2021. ([Paper](https://www.sciencedirect.com/science/article/abs/pii/S0093641321000987))

### Interpretable and Explainable Machine Learning

1. Classically studied coherent structures only paint a partial picture of wall-bounded turbulence, 2024. ([arXiv](https://arxiv.org/abs/2410.23189))

2. Explaining wall-bounded turbulence through deep learning. 2023. ([arXiv](https://arxiv.org/abs/2302.01250))

3. Multiscale Graph Neural Network Autoencoders for Interpretable Scientific Machine Learning, 2023 ([arXiv](https://arxiv.org/abs/2302.06186) | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0021999123006320))

4. Feature importance in neural networks as a means of interpretation for data-driven turbulence models, 2023. ([Open Access Paper](https://www.sciencedirect.com/science/article/pii/S0045793023002189))

5. Interpretable A-posteriori Error Indication for Graph Neural Network Surrogate Models, 2023. ([arXiv](https://arxiv.org/abs/2311.07548) | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0045782524007631))

6. Interpreted machine learning in fluid dynamics: explaining relaminarisation events in wall-bounded shear flows, 2022. ([Paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/interpreted-machine-learning-in-fluid-dynamics-explaining-relaminarisation-events-in-wallbounded-shear-flows/C2CA43557475FF09B2FCEC06D99BB0FE) | [Data](https://datashare.ed.ac.uk/handle/10283/4424))

7. An interpretable framework of data-driven turbulence modeling using deep neural networks, 2021. ([Paper](https://aip.scitation.org/doi/10.1063/5.0048909))

8. Extracting Interpretable Physical Parameters from Spatiotemporal Systems using Unsupervised Learning, 2020. ([arXiv](https://arxiv.org/abs/1907.06011) | [Blog](https://peterparity.github.io/projects/pde_vae/))

### Physics-informed ML

1. Mitigating ill-conditioning of Reynolds-averaged Navier–Stokes equations for experimental data-driven turbulence closures, 2025. ([Paper](https://pubs.aip.org/aip/pof/article-abstract/37/4/045155/3344393/Mitigating-ill-conditioning-of-Reynolds-averaged?redirectedFrom=fulltext))

2. Turbulence closure with small, local neural networks: Forced two-dimensional and $\beta$-plane flows, 2024. ([Paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS003795?af=R) | [arXiv](https://arxiv.org/abs/2304.05029))

3. Data-driven discovery of turbulent flow equations using physics-informed neural networks, 2024. ([Paper](https://pubs.aip.org/aip/pof/article-abstract/36/3/035107/3268437/Data-driven-discovery-of-turbulent-flow-equations?redirectedFrom=fulltext))

4. Turbulence model augmented physics-informed neural networks for mean-flow reconstruction, 2024. ([Paper](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.9.034605) | [arXiv](https://arxiv.org/abs/2306.01065) | [Code](https://github.com/RigasLab/PINN_SA))

5. Data-Driven Turbulence Modeling Approach for Cold-Wall Hypersonic Boundary Layers, 2024. ([arXiv](https://arxiv.org/abs/2406.17446))

6. Generalized field inversion strategies for data-driven turbulence closure modeling, 2024. ([Open access paper](https://pubs.aip.org/aip/pof/article/36/10/105188/3318159/Generalized-field-inversion-strategies-for-data))

7. A Physics-Informed Autoencoder-NeuralODE Framework (Phy-ChemNODE) for Learning Complex Fuel Combustion Kinetics, 2024. ([Paper](https://neurips.cc/virtual/2024/100084))

8. Extending a Physics-Informed Machine Learning Network for Superresolution Studies of Rayleigh-Bénard Convection, 2023. ([arXiv](https://arxiv.org/abs/2307.02674))

9. Machine learning for RANS turbulence modeling of variable property flows, 2023. ([arXiv](https://arxiv.org/abs/2210.15384) | [Paper](https://www.sciencedirect.com/science/article/pii/S0045793023000609))

10. A probabilistic, data-driven closure model for RANS simulations with aleatoric, model uncertainty, 2023. ([arXiv](https://arxiv.org/abs/2307.02432))

<details>
<summary>Archive</summary>

11. Physics-informed data based neural networks for two-dimensional turbulence, 2022. ([arXiv](https://arxiv.org/pdf/2203.02555.pdf) | [Paper](https://aip.scitation.org/doi/abs/10.1063/5.0090050))

12. Deep Physics Corrector: A physics enhanced deep learning architecture for solving stochastic differential equations, 2022. ([arXiv](https://arxiv.org/abs/2209.09750))

13. A Physics-informed Diffusion Model for High-fidelity Flow Field Reconstruction, 2022. ([arXiv](https://arxiv.org/abs/2211.14680))

14. A fast and accurate physics-informed neural network reduced order model with shallow masked autoencoder, 2022. ([arXiv](https://arxiv.org/abs/2009.11990) | [Paper](https://www.sciencedirect.com/science/article/pii/S0021999121007361))

15. FluxNet: a physics-informed learning-based Riemann solver for transcritical flows with non-ideal thermodynamics, 2022. ([Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4216629) | [Code](https://git.uwaterloo.ca/jc9wang/fluxnet))

16. An Improved Structured Mesh Generation Method Based on Physics-informed Neural Networks, 2022. ([arXiv](https://arxiv.org/abs/2210.09546))

17. Physics-Informed Neural Networks for Inverse Problems in Supersonic Flows, 2022. ([arXiv](https://arxiv.org/abs/2202.11821) | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0021999122004648))

18. Data-driven algebraic models of the turbulent Prandtl number for buoyancy-affected flow near a vertical surface, 2021. ([arXiv](https://arxiv.org/abs/2104.01842))

19. Convolutional Neural Network Models and Interpretability for the Anisotropic Reynolds Stress Tensor in Turbulent One-dimensional Flows, 2021. ([arXiv](https://arxiv.org/abs/2106.15757))

20. Physics-aware deep neural networks for surrogate modeling of turbulent natural convection, 2021. ([arXiv](https://arxiv.org/abs/2103.03565))

21. Learned Turbulence Modelling with Differentiable Fluid Solvers, 2021. ([arXiv](https://arxiv.org/abs/2202.06988))

22. Data-driven fractional subgrid-scale modeling for scalar turbulence: A nonlocal LES approach, 2020. ([arXiv](https://arxiv.org/abs/2012.14027))

23. A machine learning framework for LES closure terms, 2020. ([arXiv](https://arxiv.org/abs/2010.03030))

24. A neural network based shock detection and localization approach for discontinuous Galerkin methods, 2020. ([arXiv](https://arxiv.org/pdf/2001.08201.pdf))

25. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, 2019. ([Paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125))

26. Neural network models for the anisotropic Reynolds stress tensor in turbulent channel flow, 2019. ([arXiv](https://arxiv.org/abs/1909.03591))

27. From deep to physics-informed learning of turbulence: Diagnostics, 2018. ([arXiv](https://arxiv.org/abs/1810.07785))

28. Subgrid modelling for two-dimensional turbulence using neural networks, 2018. ([arXiv](https://arxiv.org/abs/1808.02983) | [Code](https://github.com/Romit-Maulik/ML_2D_Turbulence))

29. Reynolds averaged turbulence modeling using deep neural networks with embedded invariance, 2016. ([Paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/reynolds-averaged-turbulence-modelling-using-deep-neural-networks-with-embedded-invariance/0B280EEE89C74A7BF651C422F8FBD1EB))

</details>

### Reduced-Order Modeling Aided ML

1. Slim multi-scale convolutional autoencoder-based reduced-order models for interpretable features of a complex dynamical system, 2025. ([arXiv](https://arxiv.org/abs/2501.03070) | [Paper](https://pubs.aip.org/aip/aml/article/3/1/016112/3337304/Slim-multi-scale-convolutional-autoencoder-based))

2. Multi-fidelity reduced-order surrogate modeling, 2024. ([arXiv](https://arxiv.org/abs/2309.00325) | [Paper](https://royalsocietypublishing.org/doi/full/10.1098/rspa.2023.0655?af=R))

3. β-Variational autoencoders and transformers for reduced-order modeling of fluid flows, 2024. ([arXiv](https://arxiv.org/abs/2304.03571) | [Paper](https://www.nature.com/articles/s41467-024-45578-4) | [Code](https://github.com/KTH-FlowAI/beta-Variational-autoencoders-and-transformers-for-reduced-order-modelling-of-fluid-flows) | [Data](https://zenodo.org/records/10501216))

4. Shock wave prediction in transonic flow fields using domain-informed probabilistic deep learning, 2024. ([Paper](https://pubs.aip.org/aip/pof/article-abstract/36/1/016121/2932992/Shock-wave-prediction-in-transonic-flow-fields?redirectedFrom=fulltext))

5. Data-Driven Modeling for Transonic Aeroelastic Analysis, 2023. ([arXiv](https://arxiv.org/abs/2304.07046) | [Code, will be available](https://github.com/Nicola-Fonzi/pysu2DMD))

6. Predicting the wall-shear stress and wall pressure through convolutional neural networks, 2023. ([arXiv](https://arxiv.org/abs/2303.00706) | [Paper](https://www.sciencedirect.com/science/article/pii/S0142727X23000991))

7. Deep learning-based reduced order model for three-dimensional unsteady flow using mesh transformation and stitching, 2023. ([arXiv](https://arxiv.org/abs/2307.07323) | Data: Contact authors)

8. Reduced-order modeling of fluid flows with transformers, 2023. ([Paper](https://pubs.aip.org/aip/pof/article-abstract/35/5/057126/2891586/Reduced-order-modeling-of-fluid-flows-with))

9. Direct data-driven forecast of local turbulent heat flux in Rayleigh–Bénard convection, 2022. ([arXiv](https://arxiv.org/abs/2202.13129) | [Paper](https://aip.scitation.org/doi/abs/10.1063/5.0087977) | Data: Contact authors)

10. Cost function for low-dimensional manifold topology assessment, 2022. ([Paper](https://www.nature.com/articles/s41598-022-18655-1) | [Data](https://tnfworkshop.org/data-archives/pilotedjet/ch4-air/) | [Code](https://github.com/kamilazdybal/cost-function-manifold-assessment))

<details>
<summary>Archive</summary>

11. From coarse wall measurements to turbulent velocity fields with deep learning, 2021. ([arXiv](https://arxiv.org/abs/2103.07387))

12. Convolutional neural network and long short-term memory based reduced order surrogate for minimal turbulent channel flow, 2021. ([arXiv](https://arxiv.org/abs/2010.13351) | Data: Contact authors)

13. Reservoir computing model of two-dimensional turbulent convection, 2020. ([arXiv](https://arxiv.org/abs/2001.10280))

14. Reduced-order modeling of advection-dominated systems with recurrent neural networks and convolutional autoencoders, 2020. ([arXiv](https://arxiv.org/pdf/2002.00470.pdf) | [Code](https://github.com/Romit-Maulik/CAE_LSTM_ROMS))

15. Time-series learning of latent-space dynamics for reduced-order model closure, 2020. ([Paper](https://linkinghub.elsevier.com/retrieve/pii/S0167278919305536) | [Code](https://github.com/Romit-Maulik/ML_ROM_Closures))

16. Echo state network for two-dimensional turbulent moist Rayleigh-Bénard convection, 2020. ([arXiv](https://arxiv.org/abs/2101.11325))

17. DeepCFD: Efficient steady-state laminar flow approximation with deep convolutional neural networks, 2020. ([arXiv](https://arxiv.org/pdf/2004.08826.pdf) | [Code](https://github.com/mdribeiro/DeepCFD))

18. Predictions of turbulent shear flows using deep neural networks, 2019. ([arXiv](https://arxiv.org/abs/1905.03634 "Paper") | [Code](https://github.com/KTH-Nek5000/DeepTurbulence "Code"))

19. A deep learning enabler for nonintrusive reduced order modeling of fluid flows, 2019. ([arXiv](https://arxiv.org/abs/1907.04945))

</details>

### Transfer Learning

1. Non-intrusive, transferable model for coupled turbulent channel-porous media flow based upon neural networks, 2024. ([Paper](https://pubs.aip.org/aip/pof/article/36/2/025112/3262262/Non-intrusive-transferable-model-for-coupled) | Data: Contact authors)

2. Stable a posteriori LES of 2D turbulence using convolutional neural networks: Backscattering analysis and generalization to higher Re via transfer learning, 2021. ([arXiv](https://arxiv.org/abs/2102.11400))

### Generative AI

1. Inpainting Computational Fluid Dynamics with Deep Learning, 2024. ([arXiv](https://arxiv.org/abs/2402.17185))

2. Generative Adversarial Reduced Order Modeling, 2024. ([arXiv](https://arxiv.org/abs/2305.15881) | [Paper](https://www.nature.com/articles/s41598-024-54067-z) | [Code](https://github.com/dario-coscia/GAROM))

3. Three-dimensional generative adversarial networks for turbulent flow estimation from wall measurements, 2024. ([arXiv](https://arxiv.org/abs/2409.06548) | [Paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/threedimensional-generative-adversarial-networks-for-turbulent-flow-estimation-from-wall-measurements/6BD96A003A1D53325D8AC04341DC1713))

### Pattern Identification and Generation

1. Observable-augmented manifold learning for multi-source turbulent flow data, 2025. ([Paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/observableaugmented-manifold-learning-for-multisource-turbulent-flow-data/40723B28236B6DCD324A4FA484F49181) | [Code](https://github.com/kfukami/Observable-CNN-AE))

2. Machine learning-based vorticity evolution and super-resolution of homogeneous isotropic turbulence using wavelet projection, 2024. ([ResearchGate](https://www.researchgate.net/publication/378145820_Machine_learning-based_vorticity_evolution_and_super-resolution_of_homogeneous_isotropic_turbulence_using_wavelet_projection) | [Paper](https://pubs.aip.org/aip/pof/article-abstract/36/2/025120/3262840/Machine-learning-based-vorticity-evolution-and?redirectedFrom=fulltext))

3. Data-driven nonlinear turbulent flow scaling with Buckingham Pi variables, 2024. ([Paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/datadriven-nonlinear-turbulent-flow-scaling-with-buckingham-pi-variables/116D45EFF6E4231E2ACC1B819C20C708) | [arXiv](https://arxiv.org/abs/2402.17990))

4. Single-snapshot machine learning for super-resolution of turbulence, 2024. ([arXiv](https://arxiv.org/abs/2409.04923) | [Paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/singlesnapshot-machine-learning-for-superresolution-of-turbulence/1AF295B2B2A5B02C47FA97C0C28A1D7F))

5. A deep neural network architecture for reliable 3D position and size determination for Lagrangian particle tracking using a single camera, 2023. ([Open Access Paper](https://iopscience.iop.org/article/10.1088/1361-6501/ace070) | [Data](https://defocustracking.com/datasets/))

6. Sparse sensor reconstruction of vortex-impinged airfoil wake with machine learning, 2023. ([arXiv](https://arxiv.org/abs/2305.05147) | [Open Access Paper](https://link.springer.com/article/10.1007/s00162-023-00657-y))

7. Identifying regions of importance in wall-bounded turbulence through explainable deep learning, 2023. ([arXiv](https://arxiv.org/abs/2302.01250) | [Code](https://github.com/KTH-FlowAI/Identifying-regions-of-importance-in-wall-bounded-turbulence-through-explainable-deep-learning))

8. Data-driven estimation of scalar quantities from planar velocity measurements by deep learning applied to temperature in thermal convection, 2023. ([Paper](https://link.springer.com/article/10.1007/s00348-023-03736-2) | Data: Contact authors)

9. Reconstruction of three-dimensional turbulent flow structures using surface measurements for free-surface flows based on a convolutional neural network, 2023. ([Paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/reconstruction-of-threedimensional-turbulent-flow-structures-using-surface-measurements-for-freesurface-flows-based-on-a-convolutional-neural-network/DA0663B3D28F78C1056AB4737675EF7B) | [arXiv](https://arxiv.org/abs/2301.11710))

10. Unsupervised deep learning for super-resolution reconstruction of turbulence, 2020. ([arXiv](https://arxiv.org/abs/2007.15324))

<details>
<summary>Archive</summary>

11. Nonlinear mode decomposition with convolutional neural networks for fluid dynamics, 2020. ([arXiv](https://arxiv.org/abs/1906.04029))

12. Deep learning in turbulent convection networks, 2019. ([Paper](https://www.pnas.org/content/116/18/8667))

13. Time-resolved turbulent velocity field reconstruction using a long short-term memory (LSTM)-based artificial intelligence framework, 2019. ([Paper](https://aip.scitation.org/doi/10.1063/1.5111558))

</details>

### Reinforcement Learning

1. Deep Reinforcement Learning for the Management of the Wall Regeneration Cycle in Wall-Bounded Turbulent Flows, 2024. ([arXiv](https://www.arxiv.org/abs/2408.06783) | [Paper](https://link.springer.com/article/10.1007/s10494-024-00609-4) | [Code](https://github.com/gmcavallazzi/CaNS_DRL))

2. Deep reinforcement learning for turbulent drag reduction in channel flows, 2023. ([arXiv](https://arxiv.org/abs/2301.09889) | [Code](https://github.com/KTH-FlowAI/MARL-drag-reduction-in-wall-bounded-flows))

3. DRLinFluids -- An open-source Python platform coupling Deep Reinforcement Learning and OpenFOAM, 2023. ([arXiv](https://arxiv.org/abs/2205.12699) | [Paper](https://pubs.aip.org/aip/pof/article-abstract/34/8/081801/2846652/DRLinFluids-An-open-source-Python-platform-of?redirectedFrom=fulltext) | [Code](https://github.com/venturi123/DRLinFluids))

4. Automating Turbulence Modeling by Multi-Agent Reinforcement Learning, 2020. ([arXiv](https://arxiv.org/abs/2005.09023) | [Code](https://github.com/cselab/MARL_LES))

### Geometry Optimization or Generation

1. Deep reinforcement learning and mesh deformation integration for shape optimization of a single pin fin within a micro channel, 2025. ([Paper](https://www.sciencedirect.com/science/article/pii/S0017931024010718))

2. TripNet: Learning Large-scale High-fidelity 3D Car Aerodynamics with Triplane Networks, 2025. ([Paper](https://arxiv.org/abs/2503.17400))

3. Data-driven prediction of the performance of enhanced surfaces from an extensive CFD-generated parametric search space, 2023. ([Paper](https://iopscience.iop.org/article/10.1088/2632-2153/acca60) | Data: Contact authors)

4. Robust optimization of heat-transfer-enhancing microtextured surfaces based on machine learning surrogate models, 2023. ([Paper](https://www.sciencedirect.com/science/article/pii/S0735193323006073) | Data: Contact authors)

5. Deep reinforcement learning for heat exchanger shape optimization, 2022. ([Paper](https://www.sciencedirect.com/science/article/abs/pii/S001793102200583X) | [Article](https://uwaterloo.ca/research/catalogs/watco-technologies/deep-reinforcement-learning-heat-exchanger-shape))

### Others

1. Single-snapshot machine learning for turbulence super resolution, 2024. ([arXiv](https://arxiv.org/abs/2409.04923))

2. Learning Lagrangian Fluid Mechanics with E(3)-Equivariant Graph Neural Networks, 2023. ([arXiv](https://arxiv.org/abs/2305.15603) | [Code](https://github.com/tumaer/sph-hae))

3. An unsupervised machine-learning-based shock sensor for high-order supersonic flow solvers, 2023. ([arXiv](https://arxiv.org/abs/2308.00086) | [Code](https://github.com/andres-mg/2023_gmm_shock_sensor))

4. Prospects of federated machine learning in fluid dynamics, 2022. ([Paper](https://aip.scitation.org/doi/10.1063/5.0104344))

5. Graph neural network-accelerated Lagrangian fluid simulation, 2022. ([Paper](https://www.sciencedirect.com/science/article/pii/S0097849322000206))

6. Engine Combustion System Optimization Using Computational Fluid Dynamics and Machine Learning: A Methodological Approach, 2021. ([Paper](https://asmedigitalcollection.asme.org/energyresources/article-abstract/143/2/022306/1086007/Engine-Combustion-System-Optimization-Using))

7. Physics guided machine learning using simplified theories, 2021. ([Paper](https://aip.scitation.org/doi/10.1063/5.0038929) | [Code](https://github.com/surajp92/PGML))

8. Nonlinear mode decomposition with convolutional neural networks for fluid dynamics, 2020. ([arXiv](https://arxiv.org/abs/1906.04029))

9. Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations, 2020. ([Paper](https://science.sciencemag.org/content/367/6481/1026))

10. Forecasting of spatiotemporal chaotic dynamics with recurrent neural networks: a comparative study of reservoir computing and backpropagation algorithms, 2019. ([arXiv](https://arxiv.org/abs/1910.05266))

<details>
<summary>Archive</summary>

11. Data-assisted reduced-order modeling of extreme events in complex dynamical systems, 2018. ([Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0197704))

</details>

### Books

1. [Approaching machine learning problems in computational fluid dynamics and computer aided engineering applications: A Monograph for Beginners](https://www.amazon.de/-/en/Approaching-learning-computational-engineering-applications/dp/B0CZF4YN31), 2024.

## ML-Focused Events

1. [Open Conference of AI Agents for Science 2025](https://agents4science.stanford.edu), October 2025, Virtual.

2. [1st International Symposium on AI and Fluid Mechanics (AIFLUIDs 2025)](https://www.aifluids.net/abouts), May 2025, Greece.

3. [Euromech Colloquium on Data-Driven Fluid Dynamics/2nd ERCOFTAC Workshop on Machine Learning for Fluid Dynamics](https://629.euromech.org), April 2025, London, UK.

4. [4th Automotive CFD Prediction Workshop](https://autocfd.org), November 2024, Belfast, Ireland.

5. [D3S3: Data-driven and Differentiable Simulations, Surrogates, and Solvers -- Workshop @ NeurIPS '24](https://d3s3workshop.github.io), tentative 2024, Vancouver, Canada.

6. [AI and Data-driven Simulation Forum](https://simpulse.de/157/ai-sim-forum), July 2024, Stuttgart, Germany.

7. [Workshop on Machine Learning for Fluid Dynamics](https://www.ercoftac.org/events/machine-learning-for-fluid-dynamics/), March 2024, France.

8. [Machine Learning for Fluid Mechanics: Analysis, Modeling, Control and Closures](https://www.datadrivenfluidmechanics.com/), February 2024, Belgium.

9. [629 - Data-driven fluid mechanics](https://euromech.org/colloquia/colloquia-2023/629), 2024, Italy.

10. [Lecture Series on Hands on Machine Learning for Fluid Dynamics 2023](https://www.vki.ac.be/index.php/events-ls/events/eventdetail/552/-/online-on-site-hands-on-machine-learning-for-fluid-dynamics-2023), 2023, von Karman Institute, Belgium.

<details>
<summary>Archive</summary>

11. [Turbulence Modeling: Roadblocks, and the Potential for Machine Learning](https://turbmodels.larc.nasa.gov/turb-prs2021.html), 2022, USA.

12. [Mini symposia: Analysis of Real World and Industry Applications: emerging frontiers in CFD computing, machine learning and beyond](https://www.wccm2022.org/minisymposia1217.html), 2022, Yokohama, Japan.

13. [IUTAM Symposium on Data-driven modeling and optimization in fluid mechanics](https://conferences.au.dk/iutam), 2022, Denmark.

14. [33rd Parallel Computational Fluid Dynamics International Conference](https://www.ercoftac.org/events/parcfd-2022/), 2022, Italy.

15. [Workshop: data-driven methods in fluid mechanics](https://fluids.leeds.ac.uk/2022/09/02/workshop-data-driven-methods-in-fluid-mechanics/), 2022, Leeds, UK.

16. [Symposium on Model-Consistent Data-driven Turbulence Modeling](http://turbgate.engin.umich.edu/symposium/index21.html), 2021, Virtual Event.

17. [International Workshop on Data-driven Modeling and Optimization in Fluid Mechanics](https://www.istm.kit.edu/dmofm.php), 2019, Karlsruhe, Germany.

</details>

## Available Datasets

1. DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks, 2024. ([Database](https://dataverse.harvard.edu/dataverse/DrivAerNet) | [Paper-1](https://neurips.cc/virtual/2024/poster/97609) | [Paper-2](https://asmedigitalcollection.asme.org/mechanicaldesign/article-abstract/147/4/041712/1213244/DrivAerNet-A-Parametric-Car-Dataset-for-Data?redirectedFrom=fulltext) | [Video](https://www.youtube.com/watch?v=Y2-s0R_yHpo) | [Code](https://github.com/Mohamedelrefaie/DrivAerNet))

2. KTH FLOW: A rich dataset of different turbulent flows generated by DNS, LES and experiments. ([Simulation data](https://www.flow.kth.se/flow-database/simulation-data-1.791810/ "Data") | [Experimental data](https://www.flow.kth.se/flow-database/experimental-data-1.791818 "Experimental data") | [Paper-1](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/history-effects-and-near-equilibrium-in-adversepressuregradient-turbulent-boundary-layers/39C38082C380F396D004B65F438C296A "Paper-1"))

3. Johns Hopkins Turbulence Databases: High-quality datasets for different flow problems. ([Database](http://turbulence.pha.jhu.edu/webquery/query.aspx "Database") | [Paper](https://www.tandfonline.com/doi/abs/10.1080/14685248.2015.1088656?journalCode=tjot20 "Paper"))

4. Vreman Research: Turbulent channel flow dataset generated from simulation, useful for closure modeling. ([Data](http://www.vremanresearch.nl/channel.html "Data") | [Paper-1](http://www.vremanresearch.nl/Vreman_Kuerten_Chan180_PF2014.pdf "Paper-1") | [Paper-2](http://www.vremanresearch.nl/Vreman_Kuerten_Chan590_PF2014.pdf "Paper-2"))

5. CTR Stanford: Dataset for turbulent pipe flow and boundary layer generated with DNS. ([Database](https://ctr.stanford.edu/research-data "Database") | [Paper](https://www.pnas.org/content/114/27/E5292 "Paper"))

6. sCO2: Spatial data along the tube for heated and cooled pipe under supercritical pressure. It includes around 50 cases, which is a good start for regression-based models to replace correlations. ([Data](https://www.ike.uni-stuttgart.de/forschung/Ueberkritisches-CO2/dns/ "Data") | [Paper-1](https://www.sciencedirect.com/science/article/abs/pii/S0017931017353176 "Paper-1") | [Paper-2](https://www.sciencedirect.com/science/article/abs/pii/S0017931017307998 "Paper-2"))

## Online Resources

1. A first course on machine learning from Nando di Freitas: Little old, recorded in 2013 but very concise and clear. ([YouTube](https://www.youtube.com/playlist?list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6 "YouTube") | [Slides](https://www.cs.ubc.ca/~nando/540-2013/lectures.html "Slides"))

2. Steve Brunton has a wonderful channel for a variety of topics ranging from data analysis to machine learning applied to fluid mechanics. ([YouTube](https://www.youtube.com/c/Eigensteve/playlists "YouTube"))

3. Nathan Kutz has a super nice channel devoted to applied mathematics for fluid mechanics. ([YouTube](https://www.youtube.com/channel/UCoUOaSVYkTV6W4uLvxvgiFA/videos "YouTube"))

4. [Steve Brunton](https://www.youtube.com/@drneilashton) has a podcasting channel where he dive deeper on topic related to CFD and AI in general. ([YouTube](https://www.youtube.com/@drneilashton "YouTube"))

5. For beginners, a good resource to learn OpenFOAM from József Nagy. OpenFOAM can be adapted for applying ML model coupled with N-S equations (e.g. RANS/LES closure). ([YouTube](https://www.youtube.com/c/J%C3%B3zsefNagyOpenFOAMGuru/playlists "YouTube"))

6. A course on [Machine learning in computational fluid dynamics](https://github.com/AndreWeiner/ml-cfd-lecture) from TU Braunschweig.

7. Looking for courses on TensorFlow, PyTorch, GANs, etc.? Have a look at [this wonderful YouTube channel](https://www.youtube.com/c/AladdinPersson/playlists).

8. Interviews with researchers and podcasts about fluid mechanics, machine learning, and simulation are available [on this YouTube channel from Jousef Murad](https://www.youtube.com/c/TheEngiineer/videos).

9. Lecture series videos from [Data-Driven Fluid Mechanics: Combining First Principles and Machine Learning](https://www.datadrivenfluidmechanics.com/index.php/lectures-videos).

10. [Substack - AI/Machine learning in fluid mechanics, engineering, physics](https://hodgesj.substack.com) is informative with several ML-for-fluid examples and reviews.

## Blogs and News Articles

1. [Accelerating CFD development from years to weeks with agentic AI and AWS](https://aws.amazon.com/blogs/hpc/accelerating-cfd-development-from-years-to-weeks-with-agentic-ai-and-aws/), 2025. (AWS)

2. [AI for Science, Energy and Security (Special Remarks by DOE Secretary Granholm)](https://www.nvidia.com/en-us/on-demand/session/aisummitdc24-sdc1080/?playlistId=playList-57a63ddc-d012-41fa-9958-e97e775a94b4), 2024. (Panel discussion, NVIDIA)

3. [NVIDIA, Rolls-Royce and Classiq Announce Quantum Computing Breakthrough for Computational Fluid Dynamics in Jet Engines](https://nvidianews.nvidia.com/news/nvidia-rolls-royce-and-classiq-announce-quantum-computing-breakthrough-for-computational-fluid-dynamics-in-jet-engines?ncid=so-nvsh-992286&dysig_tid=d8392875c185424f99626f91fc2aa79a#cid=hpc02_so-nvsh_en-eu), 2023. (NVIDIA)

4. [Develop Physics-Informed Machine Learning Models with Graph Neural Networks](https://developer.nvidia.com/blog/develop-physics-informed-machine-learning-models-with-graph-neural-networks/), 2023. (NVIDIA)

5. [The AI algorithm reduces design cycles/costs and time-to-market for advanced products](https://www.anl.gov/taps/article/activo-software-named-finalist-for-the-2023-rd-100-awards), 2023. (ANL)

6. [Closing the gap between High-Performance Computing (HPC) and artificial intelligence (AI)](https://developer.hpe.com/blog/closing-the-gap-between-hpc-and-ai/), 2023. (HPE)

7. [Optimize F1 aerodynamic geometries via Design of Experiments and machine learning](https://aws.amazon.com/blogs/machine-learning/optimize-f1-aerodynamic-geometries-via-design-of-experiments-and-machine-learning/), 2022. (AWS)

8. [A general purpose list for transitioning to data science and ML](https://ikespand.github.io/posts/resources_ml/), 2021.

9. [A compiled list of projects from NVIDIA where AI and CFD were used](https://developer.nvidia.com/taxonomy/term/727), 2021.

10. [AI for CFD](https://becominghuman.ai/ai-for-cfd-intro-part-1-d1184936fc47), 2021. (Medium)

<details>
<summary>Archive</summary>

11. [4 Myths about AI in CFD](https://blogs.sw.siemens.com/simcenter/4-myths-about-ai-in-cfd/), 2021. (Siemens)

12. [Accelerating Product Development with Physics-Informed Neural Networks and NVIDIA Modulus](https://developer.nvidia.com/blog/accelerating-product-development-with-physics-informed-neural-networks-and-modulus/), 2021. (NVIDIA)

13. [Machine Learning in Computational Fluid Dynamics](https://towardsdatascience.com/machine-learning-in-computational-fluid-dynamics-7018941414b9), 2020. (TowardsDataScience)

14. [Studying the nature of turbulence with Neural Concept's deep learning platform](https://www.numeca.com/readnews/article/616 "Studying the nature of turbulence with Neural Concept's deep learning platform"), 2020. (Numeca)

15. [A case for machine learning in CFD](https://tinyurl.com/2f6u8jab), 2020. (Medium)

16. [Machine Learning for Accelerated Aero-Thermal Design in the Age of Electromobility](https://blog.engys.com/machine-learning-for-accelerated-aero-thermal-design-in-the-age-of-electromobility/ "Machine Learning for Accelerated Aero-Thermal Design in the Age of Electromobility"), 2020. (Engys)

17. [When CAE Meets AI: Deep Learning For CFD Simulations](https://blog.theubercloud.com/when-cae-meets-ai-deep-learning-for-cfd-simulations), 2019. (Ubercloud)

18. [Supercomputing simulations and machine learning help improve power plant](https://www.eurekalert.org/pub_releases/2018-08/gcfs-ssa082018.php "Supercomputing simulations and machine learning help improve power plant"), 2018.

19. [CFD + Machine learning for super fast simulations](https://www.reddit.com/r/CFD/comments/5n91uz/cfd_machine_learning_for_super_fast_simulations/ "CFD + Machine learning for super fast simulations"), 2017. (Reddit)

20. [What is the role of Artificial Intelligence (AI) or Machine Learning in CFD?](https://www.quora.com/What-is-the-role-of-Artificial-Intelligence-AI-or-Machine-Learning-in-CFD "What is the role of Artificial Intelligence (AI) or Machine Learning in CFD?"), 2017. (Quora)

21. [Convolutional Neural Networks for Steady Flow Approximation](https://www.autodesk.com/research/publications/convolutional-neural-networks), 2016. (Autodesk)

</details>

## Ongoing Research, Projects and Labs

1. [Center for Data-Driven Computational Physics](http://cddcp.sites.uofmhosting.net/), University of Michigan, USA.

2. [VinuesaLab](https://www.vinuesalab.com/), KTH, Sweden.

3. [DeepTurb](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-maschinenbau/profil/institute-und-fachgebiete/fachgebiet-stroemungsmechanik/carl-zeiss-stiftung-projekt-deepturb): Deep Learning in and of Turbulence, TU Ilmenau, Germany.

4. [Thuerey Group](https://ge.in.tum.de/research/): Numerical methods for physics simulations with deep learning, TU Munich, Germany.

5. [Focus Group Data-driven Dynamical Systems Analysis in Fluid Mechanics](https://www.ias.tum.de/ias/research-areas/advanced-computation-and-modeling/data-driven-dynamical-systems-analysis-in-fluid-mechanics/), TU Munich, Germany.

6. [Mechanical and AI LAB (MAIL)](https://sites.google.com/view/barati?pli=1), Carnegie Mellon University, USA.

7. [Karniadakis's CRUNCH group](https://www.brown.edu/research/projects/crunch/current-research-0), Brown University, USA.

8. [MS 6: Machine Learning and Simulation Science](https://www.simtech2023.uni-stuttgart.de/program/minisymposia/ms6/), University of Stuttgart, Germany.

9. [Special Interest Groups 54: Machine Learning for Fluid Dynamics](https://www.ercoftac.org/special_interest_groups/54-machine-learning-for-fluid-dynamics/), Europe.

10. [Fukagata Lab](https://kflab.jp/en/index.php?21H05007), Keio University, Japan.

<details>
<summary>Archive</summary>

11. [The Scalable Scientific Machine Learning Lab](https://scalable-sciml-lab.org/about/), Department of Earth Science and Engineering, Imperial College London, UK.

</details>

## Open-Source Codes, Tutorials and Examples

1. Repository [OpenFOAM Machine Learning Hackathon](https://github.com/OFDataCommittee/OFMLHackathon) has various projects originating from the [Data Driven Modelling Special Interest Group](https://wiki.openfoam.com/Data_Driven_Modelling_Special_Interest_Group).

2. Repository [machine-learning-applied-to-cfd](https://github.com/AndreWeiner/machine-learning-applied-to-cfd "machine-learning-applied-to-cfd") has some excellent examples for getting started with CFD and ML.

3. Repository [Computational-Fluid-Dynamics-Machine-Learning-Examples](https://github.com/loliverhennigh/Computational-Fluid-Dynamics-Machine-Learning-Examples "Computational-Fluid-Dynamics-Machine-Learning-Examples") has an example implementation for predicting drag from the boundary conditions alongside predicting the velocity and pressure field from the boundary conditions.

4. [Image Based CFD Using Deep Learning](https://github.com/IllusoryTime/Image-Based-CFD-Using-Deep-Learning "Image Based CFD Using Deep Learning")

5. [Deep-Flow-Prediction](https://github.com/thunil/Deep-Flow-Prediction) has the code for data generation, neural network training, and evaluation.

6. [TensorFlowFoam](https://github.com/argonne-lcf/TensorFlowFoam) has a few tutorials on TensorFlow and OpenFOAM.

7. [Reduced-order modeling of reacting flows using data-driven approaches](https://github.com/kamilazdybal/ROM-of-reacting-flows-Springer) has a Jupyter Notebook example for data-driven modeling.

8. [Tutorial on the Proper Orthogonal Decomposition (POD) by Julien Weiss](https://depositonce.tu-berlin.de/bitstream/11303/9456/5/podnotes_aiaa2019.pdf "Tutorial on the **Proper Orthogonal Decomposition (POD)** by Julien Weiss"): A step-by-step tutorial including the data and a MATLAB implementation. POD is often used for dimensionality reduction.

9. [Repository from KTH-FLOW for ML in Fluid Dynamics](https://github.com/KTH-FlowAI) has several implementations from various published papers.

## Companies Focusing on ML
0. [PhysicsX] is building AI-driven simulation software stack for engineering and manufacturing across advanced industries. In its Series B funding, [company raised more than USD 155M, valuing the company at nearly USD 1B](https://www.physicsx.ai/newsroom/physicsx-announces-extension-to-series-b-round). 

1. [P-1 AI](http://p-1.ai) promises to deliver engineering AGI and [managed to raise $23 million seed financing](https://www.businesswire.com/news/home/20250425073932/en/P-1-AI-Comes-Out-of-Stealth-Aims-to-Build-Engineering-AGI-for-Physical-Systems).

2. [Emmi AI](https://www.emmi.ai) promises to deliver AI-powered physics architecture and models that unlock real-time interaction, slashing simulation times from days to seconds. [They have raised €15M in a seed round](https://techfundingnews.com/physics-meets-ai-emmi-ai-snaps-e15m-in-largest-austrian-seed-round-to-supercharge-engineering-simulations/).

3. [DeepSim](https://www.deepsim.io/technology) is a startup backed by [Y Combinator](https://www.ycombinator.com/companies/deepsim-inc). As per Crunchbase, DeepSim raised $500K on 2024-09-25 in a pre-seed round.

4. [Neural Concepts](https://neuralconcept.com/) is harnessing deep learning for accelerated simulation and design. [They have raised $27 million in a Series B round](https://www.neuralconcept.com/post/neural-concept-raises-27-million-series-b-to-further-accelerate-global-growth#:~:text=Neural%20Concept%20Raises%20%2427%20million%20Series%20B%20to%20further%20accelerate%20global%20growth,-Press%20Releases&text=%E2%80%8D%5BLausanne%2C%20Switzerland%2C%20June,a%20Series%20B%20funding%20round.).

5. [Navier AI](https://navier.ai/product) is building 1000x faster simulations using physics-ML solvers. Navier AI's fast CFD platform will enable engineers to quickly explore design spaces and perform analysis-in-the-loop design optimization. They are also backed by [Y Combinator](https://www.ycombinator.com/companies/navier-ai).

6. [ToffeeX](https://toffeex.com/) is a cloud-based, physics-driven generative design software that employs physics simulations to guide the engineering process, creating optimized designs that meet the user’s objectives.

7. [byteLAKE](https://bytelake.com/product/) offers a CFD Suite, which is a collection of AI models to [significantly accelerate the execution of CFD simulations](https://becominghuman.ai/ai-accelerated-cfd-computational-fluid-dynamics-how-does-bytelakes-cfd-suite-work-fea42fd0761e).

8. [NVIDIA](https://developer.nvidia.com/blog/modulus-v21-06-released-for-general-availability/) is leading with many products and libraries.

9. [NAVASTO](https://www.navasto.de/en/) has a few products that combine AI with CFD.

10. [Phinyx AI](https://www.phinyx.ai/solution) leverages Physics-Informed Machine Learning to extract deeper insights while reducing common challenges in traditional AI, such as AI hallucinations.

<details>
<summary>Archive</summary>

11. [Ansys AI](https://www.ansys.com/ai) is a cloud-enabled generative AI platform that can use simulation results to reliably assess the performance of a new design within minutes.

12. [Spaider AI](https://www.spaider.ai) claims to provide workflows to train neural networks, speed up and improve numerical simulations, and enable real-time performance predictions for CFD, FEM and CEM.

</details>

## Open-Source CFD Codes

The following open-source CFD codes can be adapted for synthetic data generation. Some of them can also be used for RANS/LES closure modeling based upon ML.

1. [Nek5000](https://nek5000.mcs.anl.gov/)
2. [OpenFOAM](https://www.openfoam.com/)
3. [PyFr](http://www.pyfr.org/)
4. [Nektar++](https://www.nektar.info/)
5. [Flexi](https://www.flexi-project.org/)
6. [SU2](https://su2code.github.io/)
7. [code_saturne](https://www.code-saturne.org/cms/web/)
8. [Dolfyn](https://www.dolfyn.net/)
9. [Neko](https://github.com/ExtremeFLOW/neko)
10. [Snek5000](https://snek5000.readthedocs.io/en/latest/)

## Support Forums

1. [CFDOnline](https://www.cfd-online.com/Forums/tags/machine%20learning.html)
2. [StackExchange](https://scicomp.stackexchange.com/questions/tagged/machine-learning)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ikespand/awesome-machine-learning-fluid-mechanics&type=Date)](https://www.star-history.com/#ikespand/awesome-machine-learning-fluid-mechanics&Date)

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/ikespand">
        <img src="https://github.com/ikespand.png?size=100" width="100" alt="ikespand"/><br />
        <sub><b>@ikespand</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Foroozani">
        <img src="https://github.com/Foroozani.png?size=100" width="100" alt="Foroozani"/><br />
        <sub><b>@Foroozani</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/LeoYML">
        <img src="https://github.com/LeoYML.png?size=100" width="100" alt="LeoYML"/><br />
        <sub><b>@LeoYML</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Mohamedelrefaie">
        <img src="https://github.com/Mohamedelrefaie.png?size=100" width="100" alt="Mohamedelrefaie"/><br />
        <sub><b>@Mohamedelrefaie</b></sub>
      </a>
    </td>
  </tr>
</table>
