# Generator of random fibers dataset
# Function: Preprocess Data generation
# Developed by Yijing Zhou
# 09/08/2023


import numpy as np
import random
import math


# Class for generating geometry of one RVE
class Gen_rfd():

    def __init__(self, dim_rve, num_fiber, rad_fiber, spa_ff, spa_fe):

        self.dim_rve = dim_rve
        self.num_fiber = num_fiber
        self.rad_fiber = rad_fiber
        self.spa_ff = spa_ff
        self.spa_fe = spa_fe

    def Gen(self):
        
        # Generate arrays we need
        center_v = np.zeros((self.num_fiber, 2))
        point_v = np.zeros((self.num_fiber, 2))

        # Counter
        i = 0
        i_2 = 0

        while i < self.num_fiber:

            # Random generate a point
            x = random.uniform(-self.dim_rve / 2, self.dim_rve / 2)
            y = random.uniform(-self.dim_rve / 2, self.dim_rve / 2)

            if i == 0:
                # Edge check
                if self.rad_fiber - self.dim_rve / 2 + self.spa_fe < x < -self.rad_fiber + self.dim_rve / 2 - self.spa_fe and \
                self.rad_fiber - self.dim_rve / 2 + self.spa_fe < y < -self.rad_fiber + self.dim_rve / 2 - self.spa_fe:
                    
                    center_v[i][0] = x
                    center_v[i][1] = y
                    point_v[i][0] = x + self.rad_fiber
                    point_v[i][1] = y
                    i = i + 1

            else:
                # Edge check
                if self.rad_fiber - self.dim_rve / 2 + self.spa_fe < x < -self.rad_fiber + self.dim_rve / 2 - self.spa_fe and \
                self.rad_fiber - self.dim_rve / 2 + self.spa_fe < y < -self.rad_fiber + self.dim_rve / 2 - self.spa_fe:
                    
                    # Traverse the generated center check distance
                    k = 0
                    for j in range(i):
                        dist_c = ((x - center_v[j][0]) ** 2 + (y - center_v[j][1]) ** 2) ** (1/2)
                        if dist_c > self.rad_fiber * 2 + self.spa_ff:
                            k = k + 1

                    # Overlap check
                    if k == i:
                        center_v[i][0] = x
                        center_v[i][1] = y
                        point_v[i][0] = x + self.rad_fiber
                        point_v[i][1] = y
                        i = i + 1

            # Record the number of cycles
            i_2 = i_2 + 1

            # Terminate cases where random errors may occur
            if i_2 > 100000:
                # Restart loop
                i = 0
                i_2 = 0
                center_v = np.zeros((self.num_fiber, 2))
                point_v = np.zeros((self.num_fiber, 2))

        return center_v, point_v
    

# Class for generating a file containing geometry of RVEs
class Gen_rfd_file():

    def __init__(self, num_rve, dim_rve, num_fiber, rad_fiber, spa_ff, spa_fe):

        self.num_rve = num_rve
        self.dim_rve = dim_rve
        self.num_fiber = num_fiber
        self.rad_fiber = rad_fiber
        self.spa_ff = spa_ff
        self.spa_fe = spa_fe

    def Gen_f(self):

        # Generate tensors we need
        center_dataset = np.empty((self.num_rve, self.num_fiber, 2))
        point_dataset = np.empty((self.num_rve, self.num_fiber, 2))

        # Specify class Gen_rfd
        Gen_1 = Gen_rfd(self.dim_rve, self.num_fiber, self.rad_fiber, self.spa_ff, self.spa_fe)

        # Generate dataset for any RVE
        for i in range(self.num_rve):
            center_dataset[i], point_dataset[i] = Gen_1.Gen()

        # Save datasets
        np.save("center_dataset.npy", center_dataset)
        np.save("point_dataset.npy", point_dataset)

        print("Datasets of geo_RVE has been generated")


# Class for generating single fiber of RVE (for RVE with one fiber)
class Gen_sf_file():

    def __init__(self, num_rve, dim_rve, max_sfr, min_sfr):

        self.num_rve = num_rve
        self.dim_rve = dim_rve
        self.max_sfr = max_sfr
        self.min_sfr = min_sfr

    def Gen_f(self):

        # Generate tensors we need
        rad_dataset = np.empty(self.num_rve)
        rat_dataset = np.empty(self.num_rve)

        # Generate random radius of fiber
        for i in range(self.num_rve):

            # Compute radius and volume ratio of fiber
            rad = random.uniform(self.min_sfr, self.max_sfr)
            rat = (self.dim_rve * math.pi * (rad ** 2)) / (self.dim_rve ** 3)

            rad_dataset[i] = rad
            rat_dataset[i] = rat

        # Save datasets
        np.save("f_radius_dataset.npy", rad_dataset)
        np.save("f_ratio_dataset.npy", rat_dataset)

        print("Datasets of single fiber has been generated")
