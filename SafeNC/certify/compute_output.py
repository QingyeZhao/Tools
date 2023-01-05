import numpy as np
import gurobipy as gp
from gurobipy import GRB
from certify.interval_number import IntervalNumber, interval_max, inf, sup, interval_sigmoid
from parameters import system_dim
from certify.certify_utils import mycallback


def milp_output_area(x_min, x_max, W, b, type=0):  # -1: init, 0: derivative, 1: unsafe
    weight_size = len(W)
    input_size = len(W[0])
    intval_x = []
    for i in range(0, input_size):
        t = IntervalNumber(x_min[i], x_max[i])
        intval_x.append(t)
    intval_x = np.array(intval_x).squeeze().reshape([-1, input_size])
    layer_outputs = get_layer_outputs(intval_x, W, b)
    r = layer_outputs[weight_size-1][0][0]
    c1 = inf(r)
    c2 = sup(r)
    if type == -1:
        if c2 < 0:
            return c1, 0, 0, c2, 0, 0
    elif type == 1:
        if c1 > 0:
            return c1, 0, 0, c2, 0, 0
    else:
        if c1 > 0 or c2 < 0:
            return c1, 0, 0, c2, 0, 0
    hidden_variable_num = 0
    for i in range(1, weight_size):
        hidden_variable_num = hidden_variable_num + len(W[i])
    all_variable_num = input_size + 3 * hidden_variable_num + 1
    lp_A_b_t_row = 3 * hidden_variable_num
    lp_Aeq_beq_t_row = hidden_variable_num + 1
    lp_A_Aeq = np.zeros((lp_A_b_t_row + lp_Aeq_beq_t_row, all_variable_num))
    lp_b_beq = np.zeros((lp_A_b_t_row + lp_Aeq_beq_t_row, 1)).squeeze()
    lp_A_b_loc = -1
    lp_Aeq_beq_loc = lp_A_b_t_row - 1

    binary = np.zeros((hidden_variable_num, 1), dtype=np.int).squeeze()
    binary_loc = -1

    lp_lb = np.zeros((all_variable_num, 1)).squeeze()
    lp_ub = np.ones((all_variable_num, 1)).squeeze()

    for i in range(0, input_size):
        lp_lb[i] = inf(intval_x[0][i])
        lp_ub[i] = sup(intval_x[0][i])

    lp_lb_ub_loc = input_size - 1
    process_location = input_size - 1
    y_head_record = np.zeros((1, input_size), dtype=np.int).squeeze()
    for i in range(0, input_size):
        y_head_record[i] = i

    for layer_index in range(0, weight_size - 1):
        layer_output_before_relu = layer_outputs[layer_index]
        y_head_record_last_layer = y_head_record
        y_head_record = np.zeros((1, len(W[layer_index][0])), dtype=np.int).squeeze()
        for j in range(0, len(W[layer_index][0])):
            lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
            lp_lb[lp_lb_ub_loc + 2] = max(0, inf(layer_output_before_relu[0][j]))
            lp_lb[lp_lb_ub_loc + 3] = 0
            lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])
            lp_ub[lp_lb_ub_loc + 2] = max(0, sup(layer_output_before_relu[0][j]))
            lp_ub[lp_lb_ub_loc + 3] = 1
            lp_lb_ub_loc = lp_lb_ub_loc + 3

            binary[binary_loc + 1] = process_location + 3
            binary_loc = binary_loc + 1

            lp_A_t = np.zeros((1, all_variable_num))

            lp_A_t[0][process_location + 1] = -1
            lp_A_t[0][process_location + 2] = 1
            lp_A_t[0][process_location + 3] = -inf(layer_output_before_relu[0][j])
            lp_A_Aeq[lp_A_b_loc + 1, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 1] = -inf(layer_output_before_relu[0][j])

            lp_A_t[0][process_location + 1] = 1
            lp_A_t[0][process_location + 2] = -1
            lp_A_t[0][process_location + 3] = 0
            lp_A_Aeq[lp_A_b_loc + 2, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 2] = 0

            lp_A_t[0][process_location + 1] = 0
            lp_A_t[0][process_location + 2] = 1
            lp_A_t[0][process_location + 3] = -sup(layer_output_before_relu[0][j])
            lp_A_Aeq[lp_A_b_loc + 3, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 3] = 0
            lp_A_b_loc = lp_A_b_loc + 3

            lp_Aeq_t = np.zeros((1, all_variable_num))
            for k in range(0, len(W[layer_index])):
                lp_Aeq_t[0][y_head_record_last_layer[k]] = W[layer_index][k, j]
            lp_Aeq_t[0][process_location + 1] = -1
            lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
            lp_b_beq[lp_Aeq_beq_loc + 1] = -b[layer_index][j]
            lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1

            y_head_record[j] = process_location + 2
            process_location = process_location + 3
            layer_output_before_relu[0][j] = interval_max(layer_output_before_relu[0][j], 0)

    layer_index = weight_size - 1
    layer_output_before_relu = layer_outputs[layer_index]
    y_head_record_last_layer = y_head_record
    t_process_loc = process_location

    for j in range(0, len(W[layer_index][0])):
        lp_Aeq_t = np.zeros((1, all_variable_num))
        for k in range(0, len(W[layer_index])):
            lp_Aeq_t[0][y_head_record_last_layer[k]] = W[layer_index][k, j]

        lp_Aeq_t[0][t_process_loc + 1] = -1
        lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
        lp_b_beq[lp_Aeq_beq_loc + 1] = -b[layer_index][j]
        lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
        lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
        lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])
        lp_lb_ub_loc = lp_lb_ub_loc + 1
        t_process_loc = t_process_loc + 1

    vtype = list("C" * all_variable_num)
    for i_c in range(0, len(binary)):
        vtype[binary[i_c]] = 'B'
    vtype = ''.join(vtype)

    lp_f = np.zeros((1, all_variable_num)).squeeze()
    lp_f[all_variable_num - 1] = 1

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    all_vars = []
    for i in range(0, all_variable_num):
        all_vars.append(m.addVar(lb=lp_lb[i], ub=lp_ub[i], vtype=vtype[i]))

    m.addMConstrs(A=lp_A_Aeq[0:lp_A_b_t_row], x=all_vars, sense='<', b=lp_b_beq[0:lp_A_b_t_row])
    m.addMConstrs(A=lp_A_Aeq[lp_A_b_t_row:], x=all_vars, sense='=', b=lp_b_beq[lp_A_b_t_row:])

    m.setMObjective(None, lp_f, 0.0, None, None, all_vars, GRB.MINIMIZE)
    m.optimize(mycallback)
    c1 = m.objVal
    c1_gap = m.MIPGap
    c1_v = m.getVars()
    c1_x = []
    for i in range(0, system_dim):
        c1_x.append(c1_v[i].x)

    m.setMObjective(None, lp_f, 0.0, None, None, all_vars, GRB.MAXIMIZE)
    m.optimize(mycallback)
    c2_gap = m.MIPGap
    c2 = m.objVal
    c2_v = m.getVars()
    c2_x = []
    for i in range(0, system_dim):
        c2_x.append(c2_v[i].x)

    return c1, c1_gap, c1_x, c2, c2_gap, c2_x


def get_layer_outputs(y, W, b):
    t_num_layers = len(W)

    layer_outputs = []
    layer_outputs_after_relu = []

    t_input = y

    layer_outputs.append(np.dot(t_input, W[0]) + b[0])
    layer_outputs_after_relu.append(interval_max(np.array(layer_outputs[0]), 0))

    for t_layer_index in range(1, t_num_layers):
        active_flag = np.zeros((1, len(layer_outputs[t_layer_index - 1][0]))).squeeze()
        for tj in range(0, len(layer_outputs[t_layer_index - 1][0])):
            if inf(layer_outputs[t_layer_index-1][0][tj]) >= 0:
                active_flag[tj] = 1
        layer_outputs_active = []
        layer_outputs_inactive = []
        W_active = []
        W_inactive = []
        W_mixed = []
        b_mixed = []
        for i in range(0, len(active_flag)):
            if active_flag[i] == 1:
                layer_outputs_active.append(layer_outputs_after_relu[t_layer_index - 1][0][i])
                W_active.append(W[t_layer_index][i, :]) 
                W_mixed.append(W[t_layer_index-1][:, i])  
                b_mixed.append(b[t_layer_index-1][i])  
            else:
                layer_outputs_inactive.append(layer_outputs_after_relu[t_layer_index - 1][0][i])
                W_inactive.append(W[t_layer_index][i, :])

        if sum(active_flag == 1) == 0:
            layer_outputs_inactive = np.array(layer_outputs_inactive).reshape([-1, len(layer_outputs_inactive)])
            W_inactive = np.array(W_inactive)
            output_inactive = np.dot(layer_outputs_inactive, W_inactive) + b[t_layer_index]
            layer_outputs.append(output_inactive)
        elif sum(active_flag == 0) == 0:
            W_mul = np.dot(np.array(W_mixed).transpose(), np.array(W_active))
            b_mul = np.dot(np.array(b_mixed).transpose(), np.array(W_active))
            if t_layer_index == 1:
                output_active = np.dot(t_input, W_mul) + b_mul
            else:
                output_active = np.dot(layer_outputs_after_relu[t_layer_index - 2], W_mul) + b_mul
            layer_outputs.append(output_active)
        else:
            layer_outputs_inactive = np.array(layer_outputs_inactive).reshape([-1, len(layer_outputs_inactive)])
            W_inactive = np.array(W_inactive)
            output_inactive = np.dot(layer_outputs_inactive, W_inactive) + b[t_layer_index]
            W_mul = np.dot(np.array(W_mixed).transpose(), np.array(W_active))
            b_mul = np.dot(np.array(b_mixed).transpose(), np.array(W_active))
            if t_layer_index == 1:
                output_active = np.dot(t_input, W_mul) + b_mul
            else:
                output_active = np.dot(layer_outputs_after_relu[t_layer_index - 2], W_mul) + b_mul
            layer_outputs.append(output_inactive + output_active)
        layer_outputs_after_relu.append(interval_max(np.array(layer_outputs[t_layer_index]), 0))

    return layer_outputs


def get_layer_outputs_sigmoid(y, W, b):
    t_num_layers = len(W)

    layer_outputs = []
    layer_outputs_after_sigmoid = []

    t_input = y

    layer_outputs.append(np.dot(t_input, W[0]) + b[0])
    layer_outputs_after_sigmoid.append(interval_sigmoid(np.array(layer_outputs[0])))

    for t_layer_index in range(1, t_num_layers):
        layer_outputs.append(np.dot(layer_outputs_after_sigmoid[t_layer_index - 1], W[t_layer_index]) + b[t_layer_index])
        layer_outputs_after_sigmoid.append(interval_sigmoid(np.array(layer_outputs[t_layer_index])))

    return layer_outputs






