# Preprocessing Data Generation Script
# Function: Preprocessing Data generation
# Developed by Yijing Zhou
# 10/15/2023


import numpy as np
import random
import matplotlib.pyplot as plt

from Generator_mat_properties import Gen_matfi_file
from Generator_mat_properties import Gen_matma_file
from Generator_sampling import Gen_sam
from Generator_geometry_dataset import Gen_rfd_file
from Generator_geometry_dataset import Gen_sf_file


## 0.1 Parameters

# 0.1.1 Parameters Comments

# num_rve: number of RVE models
# dim_rve: dimension of RVE models
# num_fiber: number of fibers
# rad_fiber: radius of fibers
# spa_ff: space among fibers
# spa_fe: space between fibers and the edge of RVE model
# max_sfr: maximum value of single fiber radius (for RVE with one fiber)
# min_sfr: minimum value of single fiber radius (or RVE with one fiber)

# E_ran_fiber: Range of Young's modulus of fibers
# v_fiber: Poisson's ratio of fibers
# E_ran_matrix: Range of Young's modulus of matrix
# v_matrix: Range of Poisson's ratio of matrix
# yield_p_matrix: Range of Yield point of matrix
# hard_matrix: Range of Hardening coefficient of matrix
# strain_max: Maximum strain before material fracture

# num_series: number of series
# num_time_steps: number of time steps
# size_inc: size of increments of strain of sampling
# num_pl_ser: number of series containing enough plastic points (Generally set to be equal to num_rve)
# min_pl_poi: min number of points in plastic range of a series
# beta: Parameter of J2 plasticity

# size_mesh: size of mech in ABAQUS
# ini_Inc: initial increment in ABAQUS

# 0.1.2 Value of Parameters

# RVE geometry
num_rve = 10000
dim_rve = 1.0
num_fiber = 10
rad_fiber = 0.1
spa_ff = 0.02
spa_fe = 0.02
max_sfr = 0.4
min_sfr = 0.3

# Material properties
E_ran_fiber = [324000, 324000]
v_fiber = 0.1
E_ran_matrix = [71700, 71700]
v_ran_matrix = [0.33, 0.33]
yield_p_ran_matrix = [455, 455]
hard_ran_matrix = [0.0132, 0.0132]
strain_max = 0.0216

# Properties of series
num_series = 10
num_time_steps = 50
size_inc = 0.002
num_pl_ser = 10000
min_pl_poi = 24
beta = 1

# Parameters of ABAQUS
size_mesh = 0.07
ini_Inc = 0.1


## 0.2 Task selection

print("Task selection:")
print("(Input ON or OFF)")
print("Material Properties Generation = ")
task_1 = input()
print("Series Generation = ")
task_2 = input()
print("Random Fibers Dataset Generation = ")
task_3 = input()
print("Single Fiber Dataset Generation = ")
task_4 = input()
print()


## 1. Generate material properties

if task_1 == "ON":

    Gen_fi_1 = Gen_matfi_file(E_ran_fiber, v_fiber, num_series)
    Gen_ma_1 = Gen_matma_file(E_ran_matrix, v_ran_matrix, yield_p_ran_matrix, hard_ran_matrix, num_series)

    Gen_fi_1.Gen_f_fi()
    Gen_ma_1.Gen_f_ma()


## 2. Generate series

if task_2 == "ON":

    fiber_mp_dataset = np.loadtxt("fiber_mp_dataset.txt")
    matrix_mp_dataset = np.loadtxt("matrix_mp_dataset.txt")
    mat_pro_dataset = matrix_mp_dataset

    Gen_sam_1 = Gen_sam(num_time_steps, size_inc, mat_pro_dataset, num_series, num_pl_ser, min_pl_poi, beta, strain_max)

    Gen_sam_1.Gen_f()

    # Plot and check
    strainh_dataset = np.load("strain_hat_dataset.npy")

    t_step = np.empty((50))
    for i in range(50):
        t_step[i] = i + 1
    strainh_11 = strainh_dataset[0, :, 0]
    strainh_22 = strainh_dataset[0, :, 1]
    strainh_33 = strainh_dataset[0, :, 2]
    strainh_12 = strainh_dataset[0, :, 3]
    strainh_13 = strainh_dataset[0, :, 4]
    strainh_23 = strainh_dataset[0, :, 5]

    plt.figure(1)
    plt.plot(t_step, strainh_11, label = "strain hat 11")
    plt.plot(t_step, strainh_22, label = "strain hat 22")
    plt.plot(t_step, strainh_33, label = "strain hat 33")
    plt.plot(t_step, strainh_12, label = "strain hat 12")
    plt.plot(t_step, strainh_13, label = "strain hat 13")
    plt.plot(t_step, strainh_23, label = "strain hat 23")
    plt.xlabel("time step")
    plt.ylabel("strain hat")
    plt.legend(loc = 1)
    plt.title("Plot of strain hat series check")
    plt.savefig("fig-1")


## 3. Generate geometry

# Generate random fibers dataset
if task_3 == "ON":

    Gen_rfd_1 = Gen_rfd_file(num_rve, dim_rve, num_fiber, rad_fiber, spa_ff, spa_fe)
    Gen_rfd_1.Gen_f()

# Generate single fiber dataset
if task_4 == "ON":

    Gen_sf_1 = Gen_sf_file(num_rve, dim_rve, max_sfr, min_sfr)
    Gen_sf_1.Gen_f()
