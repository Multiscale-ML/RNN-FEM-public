# RNN-FEM

**General Introduction**

This work is developed for simulation of multiscale long fiber composite materials based on machine learning. The developed scripts aim to achieve the following functions: 
1) Preprocessing, postprocessing, and homogenization of high-fidelity Representative Volume Elements (RVE)
2) Macroscopic random-walk-based strain sampling and data generation
3) Implementation of trained Gated Recurrent Unit (GRU) neural networks in ABAQUS UMAT.

Preprocessing, postprocessing, and homogenization scripts are used to simulate the RVE model of composite materials and generate training and testing datasets. The includes scripts for material properties generation, sampling, and geometry generation, which are used for preprocessing and data generation. UMAT script is used to implement trained GRU models in commercial finite element software ABAQUS, while UVARM code is used for visualization.

Please cite the following paper:

Y. Zhou, S. J. Semnani (2025). A machine learning based multi-scale finite element framework for nonlinear composite materials, Engineering with Computers. DOI: 10.1007/s00366-025-02121-3.
