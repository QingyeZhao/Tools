import os
import random
from parameters import *
import math


def parameter_load(load_net_dir, num_hidden_layers):
    w1 = np.loadtxt(load_net_dir + "a/w1", dtype=np.float32)
    b1 = np.loadtxt(load_net_dir + "a/b1", dtype=np.float32)

    w2 = np.loadtxt(load_net_dir + "a/w2", dtype=np.float32)
    b2 = np.loadtxt(load_net_dir + "a/b2", dtype=np.float32)

    w3 = np.loadtxt(load_net_dir + "a/w3", dtype=np.float32).reshape([-1, 1])
    b3 = np.loadtxt(load_net_dir + "a/b3", dtype=np.float32).reshape([1])
    return [w1, b1, w2, b2, w3, b3]



def get_controller_parameters(load_net_dir):
    controller_structure = [2, 64, 64, 1]
    controller_parameters = parameter_load(load_net_dir, len(controller_structure) - 2)
    controller_w = []
    controller_b = []
    for i in range(0, len(controller_structure) - 1):
        controller_w.append(controller_parameters[i * 2])
        controller_b.append(controller_parameters[i * 2 + 1])

    return controller_w, controller_b


def write_point(f, x):
    for i_dim in range(0, system_dim):
        f.write(str(x[i_dim]) + " ")
    f.write("\n")


def is_unsafe(xt):
    return unsafe_min[0] <= xt[0] <= unsafe_max[0] and unsafe_min[1] <= xt[1] <= unsafe_max[1]


def is_init(xt):
    return initial_min[0] <= xt[0] <= initial_max[0] and initial_min[1] <= xt[1] <= initial_max[1]


def is_inv(xt):
    return invariant_min[0] <= xt[0] <= invariant_max[0] and invariant_min[1] <= xt[1] <= invariant_max[1]


def create_train_unsafe_data():
    with open(data_dir + "unsafe.txt", "w") as f:
        pieces = [100] * system_dim
        step_size = [0.] * system_dim
        for i_dim in range(0, system_dim):
            step_size[i_dim] = (unsafe_max[i_dim] - unsafe_min[i_dim]) / pieces[i_dim]
        x = [0.] * system_dim
        for i in range(pieces[0] + 1):
            x[0] = unsafe_min[0] + i * step_size[0]
            for j in range(pieces[1] + 1):
                x[1] = unsafe_min[1] + j * step_size[1]
                if is_unsafe(x):
                    write_point(f, x)

        random_points_num = 200000
        for i in range(0, random_points_num):
            for i_dim in range(0, system_dim):
                x[i_dim] = np.random.uniform(unsafe_min[i_dim], unsafe_max[i_dim])
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

        random_points_num = 150000
        for i in range(0, random_points_num):
            for i_dim in range(0, system_dim):
                x[i_dim] = np.random.uniform(initial_min[i_dim], initial_max[i_dim])
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

        random_points_num = 300000
        for i in range(0, random_points_num):
            for i_dim in range(0, system_dim):
                x[i_dim] = np.random.uniform(invariant_min[i_dim], invariant_max[i_dim])
            write_point(f, x)


def process_type(x, n_type):
    for it in range(0, len(x[0])):
        if n_type == 2:
            x[0][it] = math.tanh(x[0][it])
        elif n_type == 1:
            x[0][it] = 1/(1+math.exp(-x[0][it]))
        else:
            x[0][it] = max(0, x[0][it])
    return x


def get_u(x, w, b):
    x = np.dot(x, w[0]) + b[0]
    x = process_type(x, 0)

    x = np.dot(x, w[1]) + b[1]
    x = process_type(x, 0)

    x = np.dot(x, w[2]) + b[2]
    x = process_type(x, 2)

    u = x * a_bound
    return u


def update(x, w, b):
    x1 = x[0][0]
    x2 = x[0][1]

    dx1 = x2
    dx2 = -0.0025*np.cos(3*x1)+0.0015*get_u(x, w, b)
    x[0][0] = x[0][0] + dx1 * diff_step_length
    x[0][1] = x[0][1] + dx2 * diff_step_length

    return x


def simulation(f, x, w, b):
    count = 0
    x = update(x, w, b)
    count = count + 1
    while count < diff_step_count:
        if not is_inv(x[0]):
            return
        write_point(f, x[0])
        x = update(x, w, b)
        count = count + 1


def create_train_trace_data():
    controller_w, controller_b = get_controller_parameters(load_net_dir)

    with open(data_dir + "trace.txt", "w") as f:
        pieces = [10] * system_dim
        step_size = [0.] * system_dim
        for i_dim in range(0, system_dim):
            step_size[i_dim] = (initial_max[i_dim] - initial_min[i_dim]) / pieces[i_dim]
        x = np.array([0.] * system_dim).reshape([1, system_dim])
        for i in range(pieces[0] + 1):
            x[0][0] = initial_min[0] + i * step_size[0]
            for j in range(pieces[1] + 1):
                x[0][1] = initial_min[1] + j * step_size[1]
                simulation(f, x, controller_w, controller_b)

        random_points_num = 500
        for i in range(0, random_points_num):
            for i_dim in range(0, system_dim):
                x[0][i_dim] = random.uniform(initial_min[i_dim], initial_max[i_dim])
            simulation(f, x, controller_w, controller_b)


def generate_data():

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    create_train_unsafe_data()
    create_train_init_data()
    create_train_invariant_data()
    create_train_trace_data()



