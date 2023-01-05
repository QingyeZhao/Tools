import numpy as np

system_dim = 2

invariant_min = [-1.2, -0.07]
invariant_min = np.array(invariant_min).squeeze()
invariant_max = [0.6, 0.08]
invariant_max = np.array(invariant_max).squeeze()

initial_min = [-0.6, -0.]
initial_min = np.array(initial_min).squeeze()
initial_max = [-0.4, 0.]
initial_max = np.array(initial_max).squeeze()

unsafe_min = [-1.2, -0.07]
unsafe_min = np.array(unsafe_min).squeeze()
unsafe_max = [-1.1, 0.07]
unsafe_max = np.array(unsafe_max).squeeze()


data_dir = "./data/"
load_net_dir = './net/ep_310/'

verification_dim_piece = 10


default_barrier_layer_neurons = [system_dim, 50, 1]
default_barrier_type = 0

number_close_0 = 1e-9
number_close_inf = 1e9

diff_step_count = 200
diff_step_length = 0.01

a_bound = 1.
