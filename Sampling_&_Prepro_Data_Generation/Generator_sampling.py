# Generator of Sampling Process
# Function: Preprocess Data generation
# Developed by Yijing Zhou
# 10/15/2023


import numpy as np
import random


# Class of sampling
class Gen_sam():


    # 0. Constructor
    def __init__(self, num_time_steps, size_inc, mat_pro_dataset, num_series, num_pl_ser, min_pl_poi, beta, strain_max):

        self.num_time_steps = num_time_steps
        self.size_inc = size_inc
        self.mat_pro_dataset = mat_pro_dataset
        self.num_series = num_series
        self.num_pl_ser = num_pl_ser
        self.min_pl_poi = min_pl_poi
        self.beta = beta
        self.strain_max = strain_max


    # 1. Function for stress computation
    def stress_com(self, stress, inc_t, k_n, alpha_n, mat_index):
        
        # Extract material properties of this series
        E = self.mat_pro_dataset[mat_index][0]
        v = self.mat_pro_dataset[mat_index][1]
        H = (2/3) * E * self.mat_pro_dataset[mat_index][3]

        # Compute elastic stiffness matrix
        mu = E / (2*(1 + v))
        lam = v*E / ((1 + v)*(1 - 2*v))

        C_el = np.array([[2*mu+lam, lam, lam, 0, 0, 0],
                         [lam, 2*mu+lam, lam, 0, 0, 0],
                         [lam, lam, 2*mu+lam, 0, 0, 0],
                         [0, 0, 0, mu, 0, 0],
                         [0, 0, 0, 0, mu, 0],
                         [0, 0, 0, 0, 0, mu]])
        
        # Initialization elpl_state
        elpl_state = 0

        # Compute stress tr
        stress_inc_t = np.dot(C_el, inc_t)
        stress_tr = stress + stress_inc_t

        # Compute xi tr
        p = (stress_tr[0] + stress_tr[1] + stress_tr[2]) / 3
        one = np.array([1, 1, 1, 0, 0, 0])
        s_tr = stress_tr - p * one
        xi_tr = s_tr - alpha_n

        # Compute norm of xi tr
        a = np.sqrt(np.sum(xi_tr[0:3] ** 2 + 2 * xi_tr[3:6] ** 2))

        if a > k_n:

            # In plastic state
            elpl_state = 1

            # Compute parameters
            n_np1 = xi_tr / a
            del_lam = (a - k_n) / (2*mu + H)
            # Compute stress
            stress_np1 = stress_tr - 2*mu*del_lam*n_np1
            # Update parameters
            k_np1 = k_n + self.beta*H*del_lam
            alpha_np1 = alpha_n + (1 - self.beta)*H*del_lam*n_np1

        else:

            # In elastic state
            elpl_state = 0
            
            # Parameters unchanged
            stress_np1 = stress_tr
            k_np1 = k_n
            alpha_np1 = alpha_n
        
        return stress_np1, k_np1, alpha_np1, elpl_state
    

    # 2. Function of single strain hat series generation
    def Gen(self, mat_index, counter_pl_ser):

        # Initialization
        strainh_series = np.empty((self.num_time_steps, 6))
        stressh_series = np.empty((self.num_time_steps, 6))
        inc_series = np.empty((self.num_time_steps, 6))
        strain_t = np.zeros((6))
        stress_t = np.zeros((6))
        alpha_n = np.zeros((6))
        pl_check = 0

        k_n = ((2/3) ** (1/2)) * self.mat_pro_dataset[mat_index][2]

        # Index of time step
        i = 0

        # Iteration for each time step
        while i < self.num_time_steps:

            inc_t = np.zeros((6))

            i_2 = 0

            # Generate strain items
            for j in range(6):
                # Amplification range check
                if counter_pl_ser < self.num_pl_ser:
                    inc_t[j] = self.size_inc * random.uniform(-1, 1)
                elif counter_pl_ser == self.num_pl_ser:
                    inc_t[j] = self.size_inc * 3 * random.uniform(-1, 1)
                # Update and check strain_t
                strain_t[j] = strain_t[j] + inc_t[j]
                if - self.strain_max < strain_t[j] < self.strain_max:
                    i_2 = i_2 + 1

            # Determine if strain is too large
            if i_2 == 6:

                # Save this time step
                strainh_series[i][:] = strain_t
                inc_series[i][:] = inc_t

                # Determine whether this time step has entered plasticity
                stress_t, k_n, alpha_n, elpl_state = self.stress_com(stress_t, inc_t, k_n, alpha_n, mat_index)

                pl_check = pl_check + elpl_state
                stressh_series[i][:] = stress_t[:]

                # Update index
                i = i + 1

            else:

                # remove this time step
                for j in range(6):
                    strain_t[j] = strain_t[j] - inc_t[j]

        return strainh_series, stressh_series, inc_series, pl_check
    

    # 3. Function for strain hat series generation
    def Gen_f(self):

        strainh_dataset = np.empty((self.num_series, self.num_time_steps, 6))
        stressh_dataset = np.empty((self.num_series, self.num_time_steps, 6))
        inc_dataset = np.empty((self.num_series, self.num_time_steps, 6))

        # Index and counter
        i = 0
        i_2 = 0
        counter_pl_ser = 0

        # Iteration for each series
        while i < self.num_series:

            # Generate a strain hat series
            strainh_series, stressh_series, inc_series, pl_check = self.Gen(i, counter_pl_ser)

            # Check if the number of plastic points in this series is larger than minimum number of plastic points
            if counter_pl_ser < self.num_pl_ser and pl_check > self.min_pl_poi:

                # Save this series
                for j in range(self.num_time_steps):
                    strainh_dataset[i][j] = strainh_series[j][:]
                    stressh_dataset[i][j] = stressh_series[j][:]
                    inc_dataset[i][j] = inc_series[j][:]

                # Update index and plastic counter
                i = i + 1
                counter_pl_ser = counter_pl_ser + 1

            # If the number of plastic series is enough
            elif counter_pl_ser == self.num_pl_ser:

                # Save random series
                for j in range(self.num_time_steps):
                    strainh_dataset[i][j] = strainh_series[j][:]
                    stressh_dataset[i][j] = stressh_series[j][:]
                    inc_dataset[i][j] = inc_series[j][:]

                i = i + 1

            # Record the number of iterations
            i_2 = i_2 + 1
        
        # Print the number of iterations
        print("Number of Iterations: "+str(i_2))

        # Save datasets
        np.save("strain_hat_dataset.npy", strainh_dataset)
        np.save("stress_hat_dataset.npy", stressh_dataset)
        np.save("increment_dataset.npy", inc_dataset)

        print("Dataset of Sampling has been generated")
