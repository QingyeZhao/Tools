import numpy as np
from parameters import *
from verify.compute_output import milp_output_area
from verify.derivative_verification import derivative_verify
from verify.verify_utils import get_parameters


def verification():
    barrier_w, barrier_b, controller_w, controller_b = get_parameters(load_net_dir)

    # verify unsafe region
    l, l_gap, l_x, u, u_gap, u_x = milp_output_area(unsafe_min, unsafe_max, barrier_w, barrier_b, 1)
    if l > 0 and l_gap < 1:
        print("Safe verification of unsafe region.")
    else:
        print("Error net for unsafe region.")
        return False, l_x, 1

    l, l_gap, l_x, u, u_gap, u_x = milp_output_area(initial_min, initial_max, barrier_w, barrier_b, -1)
    if u < 0 and u_gap < 1:
        print("Safe verification of initial set.")
    else:
        print('Error net for initial set.')
        return False, u_x, -1

    pieces = np.zeros([1, system_dim], dtype=np.int).squeeze() + verification_dim_piece
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
            obj, obj_gap, obj_x = derivative_verify(x_min, x_max, barrier_w, barrier_b, controller_w, controller_b)
            if obj < 0 and obj_gap < 1:
                continue
            else:
                print('Error net for the derivative verification.')
                print(i, j)
                continue
                return False, obj_x, 0
        # return True, 0, 0
    print('Safe verification of the derivative for invariant area.')
    return True, 0, 0












