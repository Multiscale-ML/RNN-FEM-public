# Postprocess and Homogenization script for generating RVE with Fixed fibers
# Function: Data generation
# Developed by Yijing Zhou
# 01/03/2024


import numpy as np

from odbAccess import *


# Number of .odb files of RVEs
num_rve = 500
num_time_steps = 50

# Index of beginning series minus 1
ind_m_1 = 9000

# Running range
running_range = "9001_9500"

# Stress index
S_ind = [11, 22, 33, 12, 13, 23]

# Stress hat dataset
S_hat_dataset = np.empty((num_rve, num_time_steps, 6))

# Homogenization
# Loop over series
for i in range(9000, 9500):

    # Open odb file for every series
    odb_i = openOdb(path='Job-MSThesis-FixF1-'+str(i+1)+'.odb')

    # Loop over time steps
    for j in range(num_time_steps):

        # Define step (frame)
        Frame_j = odb_i.steps['Step-1'].frames[j+1]
        
        # Extract integral point volume data from .odb file
        int_vol = Frame_j.fieldOutputs['IVOL']
        int_vol_value = int_vol.values

        # Extract integral point stress data from .odb file
        stress = Frame_j.fieldOutputs['S']
        stress_value = stress.values

        #stress_location = stress.locations
        #print(stress_location)

        # Define data vectors
        vol_v = []
        str11_v = []
        str22_v = []
        str33_v = []
        str12_v = []
        str13_v = []
        str23_v = []
        
        # Extract integral point volume data
        for vol in int_vol_value:

            vol_v.append(vol.data)

        # Extract integral point stress data
        for str_ in stress_value:

            str11_v.append(str_.data[0])
            str22_v.append(str_.data[1])
            str33_v.append(str_.data[2])
            str12_v.append(str_.data[3])
            str13_v.append(str_.data[4])
            str23_v.append(str_.data[5])

        # Homogenization process
        vol_sum = sum(vol_v)

        len_v = len(vol_v)
        len_s = len(str11_v)
        # print(len_s)

        homo11 = 0
        homo22 = 0
        homo33 = 0
        homo12 = 0
        homo13 = 0
        homo23 = 0
        for k in range(len_s):
            homo11 = homo11 + (vol_v[k] * str11_v[k])
            homo22 = homo22 + (vol_v[k] * str22_v[k])
            homo33 = homo33 + (vol_v[k] * str33_v[k])
            homo12 = homo12 + (vol_v[k] * str12_v[k])
            homo13 = homo13 + (vol_v[k] * str13_v[k])
            homo23 = homo23 + (vol_v[k] * str23_v[k])

        S_hat_dataset[i - ind_m_1][j][0] = homo11 / vol_sum
        S_hat_dataset[i - ind_m_1][j][1] = homo22 / vol_sum
        S_hat_dataset[i - ind_m_1][j][2] = homo33 / vol_sum
        S_hat_dataset[i - ind_m_1][j][3] = homo12 / vol_sum
        S_hat_dataset[i - ind_m_1][j][4] = homo13 / vol_sum
        S_hat_dataset[i - ind_m_1][j][5] = homo23 / vol_sum

    # Close .odb file
    odb_i.close()
            
    # Print label of series
    print(i+1)

# Save dataset
np.save("S_hat_dataset_" + str(running_range) + ".npy", S_hat_dataset)

#print(S_hat_dataset)
