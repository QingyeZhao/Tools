function [ok,c1,c2]=milp_output_area(x_min,x_max)
%%
% input parameters:
% x_min: the minimum of x
% x_max: the maximum of x
% output parameters:
% ok: this area is safe and do not cross the barrier
% c1: the minimum of net output
% c2: the maximum of net output
%%
ok=0;
global W b
global count_milp
global time_milp
weight_size=size(W,2);
input_size=size(W{1},1);

%% 
% get the stable and unstable neurons number
intval_x=infsup(x_min,x_max);
[active_num,negative_num,binary_num,layer_outputs,variable_num]=get_stable_number(intval_x);  % no need
ott=layer_outputs{weight_size};
r=ott(1)-ott(2);
c1=inf(r);
c2=sup(r);
if c1>=0 || c2<=0 % no need to milp
    ok=1;
    return;
end
% disp('Get stable number');
% fprintf('active_num: %d\nnegative_num: %d\nbinary_num: %d\nall: %d\nstable probability: %f\n',...
%     sum(active_num),sum(negative_num),sum(binary_num),sum(active_num)+sum(negative_num)+sum(binary_num),...
%     (sum(active_num)+sum(negative_num))/(sum(active_num)+sum(negative_num)+sum(binary_num)));

%% 
% the variable number, the input and output layer is 1 times and hidden
% layers is 3 times, before relu y, after relu y', and relu binary t
all_variable_num=input_size;  % input layer
all_variable_num=all_variable_num+3*(sum(active_num)+sum(negative_num)+sum(binary_num));% hidden layer
all_variable_num=all_variable_num+size(W{weight_size},2);  % output layer
all_variable_num=all_variable_num+1;  % the diff of output layer neurons

%% 
% initilize lp_A, lp_b, lp_Aeq, and lp_beq
lp_A_b_t_row=3*(sum(active_num)+sum(negative_num)+sum(binary_num));
lp_Aeq_beq_t_row=0;
for layer_index=1:weight_size-1
    lp_Aeq_beq_t_row=lp_Aeq_beq_t_row+active_num(layer_index)+binary_num(layer_index)+negative_num(layer_index);
end
lp_Aeq_beq_t_row=lp_Aeq_beq_t_row+size(W{weight_size},2)+1;
% init
lp_A_Aeq=zeros(lp_A_b_t_row+lp_Aeq_beq_t_row,all_variable_num);
lp_b_beq=zeros(lp_A_b_t_row+lp_Aeq_beq_t_row,1);
% processing index of lp_A and lp_Aeq
lp_A_b_loc=0;
lp_Aeq_beq_loc=lp_A_b_t_row;

%% 
% initilize binary location variable
binary=zeros(sum(active_num)+sum(negative_num)+sum(binary_num),1);
% processing index of binary location
binary_loc=0;

%% 
% initilize lp_lb, lp_ub
lp_lb=zeros(all_variable_num,1);
lp_ub=ones(all_variable_num,1);
% set lp_lb and lp_ub value of input layer
for i=1:input_size
    lp_lb(i)=inf(intval_x(i));
    lp_ub(i)=sup(intval_x(i));
end
% processing index of lp_lb and lp_ub
lp_lb_ub_loc=input_size;
%% 
% processing index of variable
process_location=input_size;
%%
% the location of y'
y_head_record=1:input_size;

%% 
% milp encoding for hidden layer,
% process every neuron to y, y', bianry t of relu
% update lp_Aeq,lp_beq,lp_A,lp_b,lp_lb,lp_ub,binary
% undate y_head_record_last_layer,y_head_record,process_location,
% and y to y' of every layer
for layer_index=1:weight_size-1
    layer_output_before_relu=layer_outputs{layer_index};
    y_head_record_last_layer=y_head_record;  % y' of last layer
    y_head_record=zeros(1,size(W{layer_index},2));  % y' of this layer
    for j=1:size(W{layer_index},2)
        % update lp_lb,lp_ub
        lp_lb(lp_lb_ub_loc+1)=inf(layer_output_before_relu(j));
        lp_lb(lp_lb_ub_loc+2)=max(0,inf(layer_output_before_relu(j)));
        lp_lb(lp_lb_ub_loc+3)=0;
        lp_ub(lp_lb_ub_loc+1)=sup(layer_output_before_relu(j));
        lp_ub(lp_lb_ub_loc+2)=max(0,sup(layer_output_before_relu(j)));
        lp_ub(lp_lb_ub_loc+3)=1;
        lp_lb_ub_loc=lp_lb_ub_loc+3;
        
        % update binary location
        binary(binary_loc+1)=process_location+3;
        binary_loc=binary_loc+1;
        
        % update lp_A,lp_b_beq
        lp_A_t=zeros(1,all_variable_num);
        
        % constrain 1
        lp_A_t(process_location+1)=-1;
        lp_A_t(process_location+2)=1;
        lp_A_t(process_location+3)=-inf(layer_output_before_relu(j));
        lp_A_Aeq(lp_A_b_loc+1,:)=lp_A_t;
        lp_b_beq(lp_A_b_loc+1)=-inf(layer_output_before_relu(j));
        
        % constrain 2
        lp_A_t(process_location+1)=1;
        lp_A_t(process_location+2)=-1;
        lp_A_t(process_location+3)=0;
        lp_A_Aeq(lp_A_b_loc+2,:)=lp_A_t;
        lp_b_beq(lp_A_b_loc+2)=0;
        
        % constrain 3
        lp_A_t(process_location+1)=0;
        lp_A_t(process_location+2)=1;
        lp_A_t(process_location+3)=-sup(layer_output_before_relu(j));
        lp_A_Aeq(lp_A_b_loc+3,:)=lp_A_t;
        lp_b_beq(lp_A_b_loc+3)=0;
        lp_A_b_loc=lp_A_b_loc+3;
        
        % update lp_Aeq,lp_beq
        lp_Aeq_t=zeros(1,all_variable_num);
        for k=1:size(W{layer_index},1)
            if y_head_record_last_layer(k)~=0
                lp_Aeq_t(y_head_record_last_layer(k))=W{layer_index}(k,j);
            end
        end
        lp_Aeq_t(process_location+1)=-1;
        lp_A_Aeq(lp_Aeq_beq_loc+1,:)=lp_Aeq_t;
        lp_b_beq(lp_Aeq_beq_loc+1)=-b{layer_index}(j);
        lp_Aeq_beq_loc=lp_Aeq_beq_loc+1;
        
        % update y_head_record,process_location, and y to y' of this layer
        y_head_record(j)=process_location+2;
        process_location=process_location+3;
        layer_output_before_relu(j)=infsup(max(0,inf(layer_output_before_relu(j))),max(0,sup(layer_output_before_relu(j))));
    end 
end

%% 
% encoding for output layer
layer_index=weight_size;
layer_output_before_relu=layer_outputs{layer_index};
y_head_record_last_layer=y_head_record;
t_process_loc=process_location; % process_location is now before output
for j=1:size(W{layer_index},2)
    %lp_Aeq,lp_beq
    lp_Aeq_t=zeros(1,all_variable_num);
    for k=1:size(W{layer_index},1)
        if y_head_record_last_layer(k)~=0
            lp_Aeq_t(y_head_record_last_layer(k))=W{layer_index}(k,j);
        end
    end
    lp_Aeq_t(t_process_loc+1)=-1;
    lp_A_Aeq(lp_Aeq_beq_loc+1,:)=lp_Aeq_t;
    lp_b_beq(lp_Aeq_beq_loc+1)=-b{layer_index}(j);
    lp_Aeq_beq_loc=lp_Aeq_beq_loc+1;
    lp_lb(lp_lb_ub_loc+1)=inf(layer_output_before_relu(j));
    lp_ub(lp_lb_ub_loc+1)=sup(layer_output_before_relu(j));
    lp_lb_ub_loc=lp_lb_ub_loc+1;
    t_process_loc=t_process_loc+1;
end 

%% 
% set r=output(1)-output(2)
lp_Aeq_t=zeros(1,all_variable_num);
lp_Aeq_t(all_variable_num)=-1;
lp_Aeq_t(all_variable_num-1)=-1;
lp_Aeq_t(all_variable_num-2)=1;
lp_A_Aeq(lp_Aeq_beq_loc+1,:)=lp_Aeq_t;
lp_b_beq(lp_Aeq_beq_loc+1)=0;
lp_Aeq_beq_loc=lp_Aeq_beq_loc+1;
lp_lb(lp_lb_ub_loc+1)=inf(r);
lp_ub(lp_lb_ub_loc+1)=sup(r);
lp_lb_ub_loc=lp_lb_ub_loc+1;

%% 
% the type of variable
vtype=blanks(all_variable_num);
for i_c=1:all_variable_num
    vtype(i_c)='C';
end
for i_c=1:size(binary,1)
    vtype(binary(i_c))='B';
end

%% 
% the symble of constrains
sense=blanks(lp_A_b_t_row+lp_Aeq_beq_t_row);
for i_c=1:lp_A_b_t_row
    sense(i_c)='<';
end
for i_c=1:lp_Aeq_beq_t_row
    sense(i_c+lp_A_b_t_row)='=';
end

%% 
% the object of programming, all is 0 except r is 1
lp_f=zeros(1,all_variable_num);
lp_f(all_variable_num)=1;

%%
% set groubi model
model.obj=lp_f;
model.A=sparse(lp_A_Aeq);
model.sense=sense;
model.rhs=lp_b_beq;
model.lb=lp_lb;
model.ub=lp_ub;
model.vtype=vtype;

% the parameters of model
params.TimeLimit=100;
params.IntFeasTol=1e-9;
params.MIPGap=1e-9;
params.OutputFlag=0;

%%
% minimum of output
try
    result = gurobi(model, params);
    count_milp=count_milp+1;
    time_milp=time_milp+result.runtime;
catch ME
    ok=0;
    return;
end
if ~isfield(result,'x')
    ok=0;
    return;
end
c1=result.objval;
%%
% maximum of output
lp_f(all_variable_num)=-1;
model.obj=lp_f;
try
    result = gurobi(model, params);
    count_milp=count_milp+1;
    time_milp=time_milp+result.runtime;
catch ME
    ok=0;
    return;
end
if ~isfield(result,'x')
    ok=0;
    return;
end
c2=-result.objval;
%%
% safe area?
if c1<0 && c2>0
    ok=0;
else
    ok=1;
end

%%
    function [active_num,negative_num,binary_num,layer_outputs,variable_num]=get_stable_number(y)
        t_num_layers=size(W,2);
        % the recode of hidden layer
        negative_num=zeros(1,t_num_layers-1);
        active_num=zeros(1,t_num_layers-1);
        binary_num=zeros(1,t_num_layers-1);
        variable_num=zeros(1,t_num_layers-1);
        
        layer_outputs=cell(1,t_num_layers);  % before relu
        layer_outputs_after_relu=cell(1,t_num_layers);
        
        % connect this layer with all before layers
        t_input=y;
        %% 
        % begin computing
        t_layer_index=1;
        layer_outputs{t_layer_index}=t_input*W{t_layer_index}+b{t_layer_index};
        layer_outputs_after_relu{t_layer_index}=max(layer_outputs{t_layer_index},0);
        t_layer_negative_num=0;
        t_layer_active_num=0;
        t_layer_binary_num=0;
        layer_variable_num_length=0;
        for tj=1:size(layer_outputs{t_layer_index},2)
            if sup(layer_outputs{t_layer_index}(tj))<=0
                t_layer_negative_num=t_layer_negative_num+1;
                layer_variable_num_length=layer_variable_num_length-3;
            elseif inf(layer_outputs{t_layer_index}(tj))>=0
                t_layer_active_num=t_layer_active_num+1;
                layer_variable_num_length=layer_variable_num_length-2;
            else
                t_layer_binary_num=t_layer_binary_num+1;
            end
            layer_variable_num_length=layer_variable_num_length+3;
        end
        negative_num(t_layer_index)=t_layer_negative_num;
        active_num(t_layer_index)=t_layer_active_num;
        binary_num(t_layer_index)=t_layer_binary_num;
        variable_num(t_layer_index)=layer_variable_num_length;
        for t_layer_index=2:t_num_layers
            % first confirm the active neurons
            active_flag=zeros(1,size(layer_outputs{t_layer_index-1},2));
            for tj=1:size(layer_outputs{t_layer_index-1},2)
                if inf(layer_outputs{t_layer_index-1}(tj))>=0
                    active_flag(tj)=1;
                end
            end
            
            % separate  computing active and inactive neurons
            layer_outputs_active=layer_outputs_after_relu{t_layer_index-1}(active_flag==1);
            layer_outputs_unactive=layer_outputs_after_relu{t_layer_index-1}(active_flag==0);
            W_active=W{t_layer_index}(active_flag==1,:);
            W_unactive=W{t_layer_index}(active_flag==0,:);
            % compute inactive neurons
            output_unactive=layer_outputs_unactive*W_unactive+b{t_layer_index};
            % compute active neurons
            W_mixed=W{t_layer_index-1}(:,active_flag==1);
            W_mul=W_mixed*W_active;
            b_mixed=b{t_layer_index-1}(active_flag==1);
            b_mul=b_mixed*W_active;
            if t_layer_index==2  % input layer, special case as it is not in layer_outputs
                output_active=t_input*W_mul+b_mul;
            else
                output_active=layer_outputs_after_relu{t_layer_index-2}*W_mul+b_mul;
            end
            layer_outputs{t_layer_index}=output_unactive+output_active;
            layer_outputs_after_relu{t_layer_index}=max(layer_outputs{t_layer_index},0);
            if t_layer_index<t_num_layers
                t_layer_negative_num=0;
                t_layer_active_num=0;
                t_layer_binary_num=0;
                layer_variable_num_length=0;
                for tj=1:size(layer_outputs{t_layer_index},2)
                    if sup(layer_outputs{t_layer_index}(tj))<=0
                        t_layer_negative_num=t_layer_negative_num+1;
                        layer_variable_num_length=layer_variable_num_length-3;
                    elseif inf(layer_outputs{t_layer_index}(tj))>=0
                        t_layer_active_num=t_layer_active_num+1;
                        layer_variable_num_length=layer_variable_num_length-2;
                    else
                        t_layer_binary_num=t_layer_binary_num+1;
                    end
                    layer_variable_num_length=layer_variable_num_length+3;
                end
                negative_num(t_layer_index)=t_layer_negative_num;
                active_num(t_layer_index)=t_layer_active_num;
                binary_num(t_layer_index)=t_layer_binary_num;
                variable_num(t_layer_index)=layer_variable_num_length;
            end
        end
    end
end