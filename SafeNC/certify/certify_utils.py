import numpy as np
from gurobipy import GRB


def parameter_load(load_net_dir, sub_net_dir_name, num_hidden_layers):
    w1 = np.loadtxt(load_net_dir + sub_net_dir_name + "/w1", dtype=np.float32)
    b1 = np.loadtxt(load_net_dir + sub_net_dir_name + "/b1", dtype=np.float32)
    if num_hidden_layers == 1:
        w2 = np.loadtxt(load_net_dir + sub_net_dir_name + "/w2", dtype=np.float32).reshape([-1, 1])
        b2 = np.loadtxt(load_net_dir + sub_net_dir_name + "/b2", dtype=np.float32).reshape([1])
        return [w1, b1, w2, b2]
    else:
        w2 = np.loadtxt(load_net_dir + sub_net_dir_name + "/w2", dtype=np.float32)
        b2 = np.loadtxt(load_net_dir + sub_net_dir_name + "/b2", dtype=np.float32)
        if num_hidden_layers == 2:
            w3 = np.loadtxt(load_net_dir + sub_net_dir_name + "/w3", dtype=np.float32).reshape([-1, 1])
            b3 = np.loadtxt(load_net_dir + sub_net_dir_name + "/b3", dtype=np.float32).reshape([1])
            return [w1, b1, w2, b2, w3, b3]
        else:
            w3 = np.loadtxt(load_net_dir + sub_net_dir_name + "/w3", dtype=np.float32)
            b3 = np.loadtxt(load_net_dir + sub_net_dir_name + "/b3", dtype=np.float32)
            if num_hidden_layers == 3:
                w4 = np.loadtxt(load_net_dir + sub_net_dir_name + "/w4", dtype=np.float32).reshape([-1, 1])
                b4 = np.loadtxt(load_net_dir + sub_net_dir_name + "/b4", dtype=np.float32).reshape([1])
                return [w1, b1, w2, b2, w3, b3, w4, b4]
            else:
                w4 = np.loadtxt(load_net_dir + sub_net_dir_name + "/w4", dtype=np.float32)
                b4 = np.loadtxt(load_net_dir + sub_net_dir_name + "/b4", dtype=np.float32)
                if num_hidden_layers == 4:
                    w5 = np.loadtxt(load_net_dir + sub_net_dir_name + "/w5", dtype=np.float32).reshape([-1, 1])
                    b5 = np.loadtxt(load_net_dir + sub_net_dir_name + "/b5", dtype=np.float32).reshape([1])
                    return [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5]
                else:
                    w5 = np.loadtxt(load_net_dir + sub_net_dir_name + "/w5", dtype=np.float32)
                    b5 = np.loadtxt(load_net_dir + sub_net_dir_name + "/b5", dtype=np.float32)
                    if num_hidden_layers == 5:
                        w6 = np.loadtxt(load_net_dir + sub_net_dir_name + "/w6", dtype=np.float32).reshape([-1, 1])
                        b6 = np.loadtxt(load_net_dir + sub_net_dir_name + "/b6", dtype=np.float32).reshape([1])
                        return [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6]
                    else:
                        w6 = np.loadtxt(load_net_dir + sub_net_dir_name + "/w6", dtype=np.float32)
                        b6 = np.loadtxt(load_net_dir + sub_net_dir_name + "/b6", dtype=np.float32)
                        if num_hidden_layers == 6:
                            w7 = np.loadtxt(load_net_dir + sub_net_dir_name + "/w7", dtype=np.float32).reshape([-1, 1])
                            b7 = np.loadtxt(load_net_dir + sub_net_dir_name + "/b7", dtype=np.float32).reshape([1])
                            return [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w7, b7]
                        else:
                            print("Not implement!\n")
                            exit(-1)


def get_parameters(load_net_dir):
    barrier_structure = np.loadtxt(load_net_dir + "barrier_structure").squeeze()
    barrier_parameters = parameter_load(load_net_dir, "barrier", len(barrier_structure) - 2)

    barrier_w = []
    barrier_b = []
    for i in range(0, len(barrier_structure) - 1):
        barrier_w.append(barrier_parameters[i * 2])
        barrier_b.append(barrier_parameters[i * 2 + 1])

    controller_structure = np.loadtxt(load_net_dir + "controller_structure")
    controller_parameters = parameter_load(load_net_dir, "controller", len(controller_structure) - 2)
    controller_w = []
    controller_b = []
    for i in range(0, len(controller_structure) - 1):
        controller_w.append(controller_parameters[i * 2])
        controller_b.append(controller_parameters[i * 2 + 1])

    return barrier_w, barrier_b, controller_w, controller_b


def mycallback(model, where):
    if where == GRB.Callback.MIP:
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        if abs(objbst - objbnd) < 0.99 * abs(objbst):
            model.terminate()







