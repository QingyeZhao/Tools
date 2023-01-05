import numpy as np


system_dim = 2

invariant_min = [-6., -2.2]
invariant_min = np.array(invariant_min).squeeze()
invariant_max = [6., 2.2]
invariant_max = np.array(invariant_max).squeeze()

initial_min = [-1., -0.2]
initial_min = np.array(initial_min).squeeze()
initial_max = [1., 0.2]
initial_max = np.array(initial_max).squeeze()

unsafe_min = [-5., -1.57]
unsafe_min = np.array(unsafe_min).squeeze()
unsafe_max = [5., 1.57]
unsafe_max = np.array(unsafe_max).squeeze()

data_dir = "./data/"
load_net_dir = './net/'

certification_dim_piece = 4

default_controller_layer_neurons = [system_dim, 10, 1]
default_barrier_layer_neurons = [system_dim, 10, 1]


default_controller_type = 1
default_barrier_type = 0

number_close_0 = 1e-9
number_close_inf = 1e4


