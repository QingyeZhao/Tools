import os
import random
from parameters import *


def write_point(f, x):
    for i_dim in range(0, system_dim):
        f.write(str(x[i_dim]) + " ")
    f.write("\n")


def is_unsafe(xt):
    return not (unsafe_min[0] <= xt[0] <= unsafe_max[0] and unsafe_min[1] <= xt[1] <= unsafe_max[1])


def is_init(xt):
    return initial_min[0] <= xt[0] <= initial_max[0] and initial_min[1] <= xt[1] <= initial_max[1]


def is_inv(xt):
    return invariant_min[0] <= xt[0] <= invariant_max[0] and invariant_min[1] <= xt[1] <= invariant_max[1]


def create_train_unsafe_data():
    with open(data_dir + "unsafe.txt", "w") as f:
        pieces = [100] * system_dim
        step_size = [0.] * system_dim
        for i_dim in range(0, system_dim):
            step_size[i_dim] = (invariant_max[i_dim] - invariant_min[i_dim]) / pieces[i_dim]
        x = [0.] * system_dim
        for i in range(pieces[0] + 1):
            x[0] = invariant_min[0] + i * step_size[0]
            for j in range(pieces[1] + 1):
                x[1] = invariant_min[1] + j * step_size[1]
                if is_unsafe(x):
                    write_point(f, x)

        random_points_num = 10000
        for i in range(0, random_points_num):
            for i_dim in range(0, system_dim):
                x[i_dim] = random.uniform(invariant_min[i_dim], invariant_max[i_dim])
            if is_unsafe(x):
                write_point(f, x)


def create_train_init_data():
    with open(data_dir + "init.txt", "w") as f:
        pieces = [100] * system_dim
        step_size = [0.] * system_dim
        for i_dim in range(0, system_dim):
            step_size[i_dim] = (initial_max[i_dim] - initial_min[i_dim]) / pieces[i_dim]
        x = [0.] * system_dim
        for i in range(pieces[0] + 1):
            x[0] = initial_min[0] + i * step_size[0]
            for j in range(pieces[1] + 1):
                x[1] = initial_min[1] + j * step_size[1]
                write_point(f, x)

        random_points_num = 10000
        for i in range(0, random_points_num):
            for i_dim in range(0, system_dim):
                x[i_dim] = random.uniform(initial_min[i_dim], initial_max[i_dim])
            write_point(f, x)


def create_train_invariant_data():
    with open(data_dir + "inv.txt", "w") as f:
        pieces = [100] * system_dim
        step_size = [0.] * system_dim
        for i_dim in range(0, system_dim):
            step_size[i_dim] = (invariant_max[i_dim] - invariant_min[i_dim]) / pieces[i_dim]
        x = [0.] * system_dim
        for i in range(pieces[0] + 1):
            x[0] = invariant_min[0] + i * step_size[0]
            for j in range(pieces[1] + 1):
                x[1] = invariant_min[1] + j * step_size[1]
                write_point(f, x)

        random_points_num = 20000
        for i in range(0, random_points_num):
            for i_dim in range(0, system_dim):
                x[i_dim] = random.uniform(invariant_min[i_dim], invariant_max[i_dim])
            write_point(f, x)


def generate_data():

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    create_train_unsafe_data()
    create_train_init_data()
    create_train_invariant_data()





