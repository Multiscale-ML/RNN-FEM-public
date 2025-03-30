# Generator of Material Properties
# Function: Preprocess Data generation
# Developed by Yijing Zhou
# 10/16/2023


import numpy as np
import random


# Class for material properties generation for fibers
class Gen_matfi_file():

    def __init__(self, E_ran_fiber, v_fiber, num_series):
        self.E_ran_fiber = E_ran_fiber
        self.v_fiber = v_fiber
        self.num_series = num_series

    def Gen_f_fi(self):
        # Define dataset of properties of fiber
        fiber_mp_dataset = np.empty((self.num_series, 2))

        # Generate properties of fiber of any series
        for i in range(self.num_series):

            fiber_mp_dataset[i][0] = random.uniform(self.E_ran_fiber[0], self.E_ran_fiber[1])
            fiber_mp_dataset[i][1] = self.v_fiber

        # Save datasets
        np.savetxt("fiber_mp_dataset.txt", fiber_mp_dataset)

        print("Dataset of fiber has been generated")


# Class for material properties generation for matrix
class Gen_matma_file():

    def __init__(self, E_ran_matrix, v_ran_matrix, yield_p_ran_matrix, hard_ran_matrix, num_series):
        self.E_ran_matrix = E_ran_matrix
        self.v_ran_matrix = v_ran_matrix
        self.yield_p_ran_matrix = yield_p_ran_matrix
        self.hard_ran_matrix = hard_ran_matrix
        self.num_series = num_series

    def Gen_f_ma(self):
        # Define dataset of properties of fiber
        matrix_mp_dataset = np.empty((self.num_series, 4))

        # Generate properties of fiber of any series
        for i in range(self.num_series):

            matrix_mp_dataset[i][0] = random.uniform(self.E_ran_matrix[0], self.E_ran_matrix[1])
            matrix_mp_dataset[i][1] = random.uniform(self.v_ran_matrix[0], self.v_ran_matrix[1])
            matrix_mp_dataset[i][2] = random.uniform(self.yield_p_ran_matrix[0], self.yield_p_ran_matrix[1])
            matrix_mp_dataset[i][3] = random.uniform(self.hard_ran_matrix[0], self.hard_ran_matrix[1])

        # Save datasets
        np.savetxt("matrix_mp_dataset.txt", matrix_mp_dataset)

        print("Dataset of matrix has been generated")
