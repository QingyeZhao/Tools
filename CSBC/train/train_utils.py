import os
from parameters import *
from train.generate_data import is_init, is_unsafe, is_inv, write_point


def get_unsafe_data():
    data = np.loadtxt(data_dir + "unsafe.txt", dtype=np.float32).reshape([-1, system_dim])
    np.random.shuffle(data)
    return data


def get_init_data():
    data = np.loadtxt(data_dir + "init.txt", dtype=np.float32).reshape([-1, system_dim])
    np.random.shuffle(data)
    return data


def get_inv_data():
    data = np.loadtxt(data_dir + "inv.txt", dtype=np.float32).reshape([-1, system_dim])
    np.random.shuffle(data)
    return data


def get_trace_data():
    data = np.loadtxt(data_dir + "trace.txt", dtype=np.float32).reshape([-1, system_dim])
    np.random.shuffle(data)
    return data


def get_data():
    data_unsafe = get_unsafe_data()
    data_init = get_init_data()
    data_inv = get_inv_data()
    data_trace = get_trace_data()

    if data_init.shape[0] >= 2 * data_unsafe.shape[0]:
        data_unsafe = data_unsafe.repeat(int(data_init.shape[0] / data_unsafe.shape[0]), axis=0)
        np.random.shuffle(data_unsafe)
    elif data_unsafe.shape[0] >= 2 * data_init.shape[0]:
        data_init = data_init.repeat(int(data_unsafe.shape[0] / data_init.shape[0]), axis=0)
        np.random.shuffle(data_init)

    return data_unsafe, data_init, data_inv, data_trace


def save_parameters(sess, barrier_saving_parameters, barrier_layer_neurons, barrier_type, save_net_dir):
    if not os.path.exists(save_net_dir):
        os.mkdir(save_net_dir)
    if not os.path.exists(save_net_dir + "barrier/"):
        os.mkdir(save_net_dir + "barrier/")

    for i in range(0, len(barrier_saving_parameters)):
        p = sess.run(barrier_saving_parameters[i])
        if i % 2 == 0:
            np.savetxt(save_net_dir + "barrier/w" + str(int(i/2) + 1), p)
        else:
            np.savetxt(save_net_dir + "barrier/b" + str(int(i/2) + 1), p)

    with open(save_net_dir + "barrier_structure", "w") as f:
        for i in range(0, len(barrier_layer_neurons)):
            f.write(str(barrier_layer_neurons[i]) + "\n")

    with open(save_net_dir + "barrier_type", "w") as f:
        f.write(str(barrier_type))


def extend_counterexample(x, dataset_flag):
    extend_num = 11
    extend_step = [0.001] * system_dim
    ori = [0.] * system_dim
    for i in range(0, len(x)):
        ori[i] = x[i] - extend_step[i] * int(extend_num/2)
    t = [0.] * system_dim

    if dataset_flag == 1:
        with open(data_dir + "unsafe.txt", "a") as f:
            for i in range(0, extend_num):
                t[0] = ori[0] + extend_step[0] * i
                for j in range(0, extend_num):
                    t[1] = ori[1] + extend_step[1] * j
                    if is_unsafe(t):
                        write_point(f, t)
    elif dataset_flag == -1:
        with open(data_dir + "init.txt", "a") as f:
            for i in range(0, extend_num):
                t[0] = ori[0] + extend_step[0] * i
                for j in range(0, extend_num):
                    t[1] = ori[1] + extend_step[1] * j
                    if is_init(t):
                        write_point(f, t)
    else:
        with open(data_dir + "inv.txt", "a") as f:
            for i in range(0, extend_num):
                t[0] = ori[0] + extend_step[0] * i
                for j in range(0, extend_num):
                    t[1] = ori[1] + extend_step[1] * j
                    if is_inv(t):
                        write_point(f, t)



