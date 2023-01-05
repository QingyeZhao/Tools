from parameters import *
from certify.compute_output import milp_output_area
from certify.derivative_certification import derivative_certify
from certify.certify_utils import get_parameters


def certification():
    barrier_w, barrier_b, controller_w, controller_b = get_parameters(load_net_dir)

    invariant_min_t = [-6., -2.2]
    invariant_min_t = np.array(invariant_min_t).squeeze()
    invariant_max_t = [6., -1.57]
    invariant_max_t = np.array(invariant_max_t).squeeze()
    l, l_gap, l_x, u, u_gap, u_x = milp_output_area(invariant_min_t, invariant_max_t, barrier_w, barrier_b, 1)
    if l > 0 and l_gap < 1:
        print("Safe certification of unsafe region1.")
    else:
        print("Error net for unsafe region1.")
        return False, l_x, 1

    invariant_min_t = [-6., 1.57]
    invariant_min_t = np.array(invariant_min_t).squeeze()
    invariant_max_t = [6., 2.2]
    invariant_max_t = np.array(invariant_max_t).squeeze()
    l, l_gap, l_x, u, u_gap, u_x = milp_output_area(invariant_min_t, invariant_max_t, barrier_w, barrier_b, 1)
    if l > 0 and l_gap < 1:
        print("Safe certification of unsafe region2.")
    else:
        print("Error net for unsafe region2.")
        return False, l_x, 1

    invariant_min_t = [-6., -1.57]
    invariant_min_t = np.array(invariant_min_t).squeeze()
    invariant_max_t = [-5., 1.57]
    invariant_max_t = np.array(invariant_max_t).squeeze()
    l, l_gap, l_x, u, u_gap, u_x = milp_output_area(invariant_min_t, invariant_max_t, barrier_w, barrier_b, 1)
    if l > 0 and l_gap < 1:
        print("Safe certification of unsafe region3.")
    else:
        print("Error net for unsafe region3.")
        return False, l_x, 1

    invariant_min_t = [5., -1.57]
    invariant_min_t = np.array(invariant_min_t).squeeze()
    invariant_max_t = [6., 1.57]
    invariant_max_t = np.array(invariant_max_t).squeeze()
    l, l_gap, l_x, u, u_gap, u_x = milp_output_area(invariant_min_t, invariant_max_t, barrier_w, barrier_b, 1)
    if l > 0 and l_gap < 1:
        print("Safe certification of unsafe region4.")
    else:
        print("Error net for unsafe region4.")
        return False, l_x, 1

    l, l_gap, l_x, u, u_gap, u_x = milp_output_area(initial_min, initial_max, barrier_w, barrier_b, -1)
    if u < 0 and u_gap < 1:
        print("Safe certification of initial set.")
    else:
        print('Error net for initial set.')
        return False, u_x, -1

    pieces = np.zeros([1, system_dim], dtype=np.int).squeeze() + certification_dim_piece
    step = (invariant_max - invariant_min)/pieces

    x_min = np.array(invariant_min)
    x_max = np.array(invariant_max)

    for i in range(0, pieces[0]):
        x_min[0] = invariant_min[0] + i * step[0]
        x_max[0] = invariant_min[0] + (i + 1) * step[0]
        for j in range(0, pieces[1]):
            x_min[1] = invariant_min[1] + j * step[1]
            x_max[1] = invariant_min[1] + (j + 1) * step[1]
            l, l_gap, l_x, u, u_gap, u_x = milp_output_area(x_min, x_max, barrier_w, barrier_b, 0)
            if (l > 0 and l_gap < 1) or (u < 0 and u_gap < 1):
                continue
            obj, obj_gap, obj_x = derivative_certify(x_min, x_max, barrier_w, barrier_b, controller_w, controller_b)
            if obj < 0 and obj_gap < 1:
                continue
            else:
                print('Error net for the derivative certification.')
                return False, obj_x, 0
    print('Safe certification of the derivative for invariant area.')
    return True, 0, 0












