# GFS-DRO
This repository contains the official code for the paper:
**Gradient Flow Sampler-based Distributionally Robust Optimization**
by Zusen Xu and Jia-Jie Zhu.

## 📖 About the Project

This project proposes a principled mathematical framework for distributionally robust optimization (DRO) using PDE gradient flows.

This gradient flow perspective allows for the construction of new, practical algorithms to sample from worst-case distributions. We introduce novel samplers based on Wasserstein Fisher-Rao (WFR) and Stein variational gradient (SVG) flows to solve Wasserstein and Sinkhorn DRO problems.

Our framework is general and provides new insights into existing methods, showing that some popular DRO algorithms can be recovered as special cases of our gradient flow framework.

## 🔬 Experiments

The repository includes the code for the numerical experiments presented in the paper. The main experiment folders correspond to the following sections:

* **`CIRCLE_CLASSIFICATION`**: Implements the robust classification on an imbalanced circle dataset (Section 6.1) .
* **`Two_moon`**: Implements the binary classification on the imbalanced 'two moons' dataset (Section 6.2). Further animations visualizing the evolution of particles are provided in two moon experiemnts. 
* **`CIFAR10_feature_regression`**: Implements the adversarial multi-class logistic regression on CIFAR-10 features (Section 6.3).
* **`uncertain_least_squares`**: Implements the robust least squares problem with parameter uncertainty (Appendix D.2).


## ✍️ Citation

If you use this code or framework in your research, please cite our paper:

> Zusen Xu and Jia-Jie Zhu. "Gradient Flow Sampler-based Distributionally Robust Optimization." Preprint, October 2025.
