# UMAT (GRU) Parameters writer (Version 2)
# Function: Write Model Parameters of GRU to UMAT
# Developed by Yijing Zhou
# 02/07/2024


import numpy as np
import os


# 1. Load Parameters

GRU_weight_ih_l0 = np.loadtxt('GRU.weight_ih_l0.txt')
GRU_weight_hh_l0 = np.loadtxt('GRU.weight_hh_l0.txt')
GRU_bias_ih_l0 = np.loadtxt('GRU.bias_ih_l0.txt')
GRU_bias_hh_l0 = np.loadtxt('GRU.bias_hh_l0.txt')

GRU_weight_ih_l1 = np.loadtxt('GRU.weight_ih_l1.txt')
GRU_weight_hh_l1 = np.loadtxt('GRU.weight_hh_l1.txt')
GRU_bias_ih_l1 = np.loadtxt('GRU.bias_ih_l1.txt')
GRU_bias_hh_l1 = np.loadtxt('GRU.bias_hh_l1.txt')

GRU_weight_ih_l2 = np.loadtxt('GRU.weight_ih_l2.txt')
GRU_weight_hh_l2 = np.loadtxt('GRU.weight_hh_l2.txt')
GRU_bias_ih_l2 = np.loadtxt('GRU.bias_ih_l2.txt')
GRU_bias_hh_l2 = np.loadtxt('GRU.bias_hh_l2.txt')

linear_weight = np.loadtxt('linear.weight.txt')
linear_bias = np.loadtxt('linear.bias.txt')

# 2. Write Parameters

# Hyper parameters
input_size = 6
hidden_size = 50
output_size = 6
num_layers = 3

if os.path.exists('UMAT_GRU_1.for'):
    os.remove('UMAT_GRU_1.for')

UMAT_GRU = open('UMAT_TD_GRU_1.for', 'r+')

lines = UMAT_GRU.readlines()

index_line = -1

# Insert parameters
for line in lines:

    index_line = index_line + 1
    
    # ------------------------------ Layer 0 ------------------------------

    # ------------------------------ Weight ------------------------------

    # w_ir_l0
    if line == '      data w_ir_l0 /\n':
        num_lines = 0
        # Insert elements
        for j in range(input_size):
            for i in range(hidden_size):
                elem_ij = GRU_weight_ih_l0[i, j]  
                # Insert final element
                if j == input_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                    index_line = index_line + 1
                # Insert other elements
                else:
                    lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                    index_line = index_line + 1
                num_lines = num_lines + 1
        # Subtract the increased number of lines (Because "lines" have added the same number of lines in this section)
        index_line = index_line - num_lines

    # w_iz_l0
    if line == '      data w_iz_l0 /\n':
        num_lines = 0
        # Insert elements
        for j in range(input_size):
            for i in range(hidden_size):
                elem_ij = GRU_weight_ih_l0[i + hidden_size, j]  
                # Insert final element
                if j == input_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                    index_line = index_line + 1
                # Insert other elements
                else:
                    lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                    index_line = index_line + 1
                num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_in_l0
    if line == '      data w_in_l0 /\n':
        num_lines = 0
        # Insert elements
        for j in range(input_size):
            for i in range(hidden_size):
                elem_ij = GRU_weight_ih_l0[i + hidden_size * 2, j]  
                # Insert final element
                if j == input_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                    index_line = index_line + 1
                # Insert other elements
                else:
                    lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                    index_line = index_line + 1
                num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_hr_l0
    if line == '      data w_hr_l0 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_hh_l0[i, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_hz_l0
    if line == '      data w_hz_l0 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_hh_l0[i + hidden_size, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_hn_l0
    if line == '      data w_hn_l0 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_hh_l0[i + hidden_size * 2, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # ------------------------------ Bias ------------------------------
        
    # b_ir_l0
    if line == '      data b_ir_l0 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_ih_l0[i]
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_iz_l0
    if line == '      data b_iz_l0 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_ih_l0[i + hidden_size]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_in_l0
    if line == '      data b_in_l0 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_ih_l0[i + hidden_size * 2]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_hr_l0
    if line == '      data b_hr_l0 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_hh_l0[i]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_hz_l0
    if line == '      data b_hz_l0 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_hh_l0[i + hidden_size]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_hn_l0
    if line == '      data b_hn_l0 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_hh_l0[i + hidden_size * 2]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # ------------------------------ Layer 1 ------------------------------

    # ------------------------------ Weight ------------------------------

    # w_ir_l1
    if line == '      data w_ir_l1 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_ih_l1[i, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_iz_l1
    if line == '      data w_iz_l1 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_ih_l1[i + hidden_size, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_in_l1
    if line == '      data w_in_l1 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_ih_l1[i + hidden_size * 2, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_hr_l1
    if line == '      data w_hr_l1 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_hh_l1[i, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_hz_l1
    if line == '      data w_hz_l1 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_hh_l1[i + hidden_size, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_hn_l1
    if line == '      data w_hn_l1 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_hh_l1[i + hidden_size * 2, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # ------------------------------ Bias ------------------------------
        
    # b_ir_l1
    if line == '      data b_ir_l1 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_ih_l1[i]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_iz_l1
    if line == '      data b_iz_l1 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_ih_l1[i + hidden_size]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_in_l1
    if line == '      data b_in_l1 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_ih_l1[i + hidden_size * 2]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_hr_l1
    if line == '      data b_hr_l1 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_hh_l1[i]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_hz_l1
    if line == '      data b_hz_l1 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_hh_l1[i + hidden_size]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_hn_l1
    if line == '      data b_hn_l1 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_hh_l1[i + hidden_size * 2]  
            # Insert final element
            if j == i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # ------------------------------ Layer 2 ------------------------------

    # ------------------------------ Weight ------------------------------

    # w_ir_l2
    if line == '      data w_ir_l2 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_ih_l2[i, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_iz_l2
    if line == '      data w_iz_l2 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_ih_l2[i + hidden_size, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_in_l2
    if line == '      data w_in_l2 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_ih_l2[i + hidden_size * 2, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_hr_l2
    if line == '      data w_hr_l2 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_hh_l2[i, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_hz_l2
    if line == '      data w_hz_l2 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_hh_l2[i + hidden_size, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # w_hn_l2
    if line == '      data w_hn_l2 /\n':
        num_lines = 0
        counter_1 = 0
        new_line = []
        # Insert elements
        for j in range(hidden_size):
            for i in range(hidden_size):
                counter_1 = counter_1 + 1
                new_line.append(round(GRU_weight_hh_l2[i + hidden_size * 2, j], 8))
                # Insert final element
                if counter_1 == 5 and j == hidden_size - 1 and i == hidden_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + '\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
                # Insert other elements
                elif counter_1 == 5:
                    lines.insert(index_line + 1, '     & ' + str(new_line[0]) + ',' + str(new_line[1]) + ',' + str(new_line[2]) + ',' + str(new_line[3]) + ',' + str(new_line[4]) + ',\n')
                    index_line = index_line + 1
                    counter_1 = 0
                    new_line = []
                    num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # ------------------------------ Bias ------------------------------
        
    # b_ir_l2
    if line == '      data b_ir_l2 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_ih_l2[i]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_iz_l2
    if line == '      data b_iz_l2 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_ih_l2[i + hidden_size]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_in_l2
    if line == '      data b_in_l2 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_ih_l2[i + hidden_size * 2]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_hr_l2
    if line == '      data b_hr_l2 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_hh_l2[i]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_hz_l2
    if line == '      data b_hz_l2 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_hh_l2[i + hidden_size]  
            # Insert final element
            if i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # b_hn_l2
    if line == '      data b_hn_l2 /\n':
        num_lines = 0
        # Insert elements
        for i in range(hidden_size):
            elem_ij = GRU_bias_hh_l2[i + hidden_size * 2]  
            # Insert final element
            if j == i == hidden_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # ------------------------------ Linear Layer ------------------------------

    # ------------------------------ Weight ------------------------------

    # w_linear
    if line == '      data w_linear /\n':
        num_lines = 0
        # Insert elements
        for j in range(hidden_size):
            for i in range(output_size):
                elem_ij = linear_weight[i, j]  
                # Insert final element
                if j == hidden_size - 1 and i == output_size - 1:
                    lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                    index_line = index_line + 1
                # Insert other elements
                else:
                    lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                    index_line = index_line + 1
                num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines

    # ------------------------------ Bias ------------------------------
        
    # b_linear
    if line == '      data b_linear /\n':
        num_lines = 0
        # Insert elements
        for i in range(output_size):
            elem_ij = linear_bias[i]  
            # Insert final element
            if i == output_size - 1:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + '\n')
                index_line = index_line + 1
            # Insert other elements
            else:
                lines.insert(index_line + 1, '     & ' + str(elem_ij) + ',\n')
                index_line = index_line + 1
            num_lines = num_lines + 1
        # Subtract the increased number of lines
        index_line = index_line - num_lines


UMAT_GRU.close()

# Write complete UMAT
UMAT_GRU_1 = open('UMAT_GRU_1.for', 'w')
UMAT_GRU_1.writelines(lines)
UMAT_GRU_1.close()
