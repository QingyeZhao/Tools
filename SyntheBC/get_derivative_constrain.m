function [r,Cons,splits,output_cons_Aeq,output_cons_beq]=get_derivative_constrain(x_min,x_max)
%%
% input parameters:
% x_min: the minimum of x
% x_max: the maximum of x
% output parameters:
% r: if the number of unstable neurons is 0, r is derivative
% Cons: a cell of tCons
%   tCons: a cell of derivative, A, and b
%   tCons.1: derivative of this piece
%   tCons.2: A of this piece
%   tCons.3: b of this piece
% splits: the recode number of pieces, 0 for the case of unstable neurons is 0
% output_cons_Aeq: Aeq of pieces
% output_cons_beq: Aeq of pieces
% this function is for 2 hidder layers net

global W b net_structure
global count_lp
global time_lp
intval_x=infsup(x_min,x_max);
input_size=size(W{1},1);
%% 
% get information of binary neurons
[active_num,negative_num,binary_num,layer_outputs,variable_num,binary_value,binary_value_zero_loc]=...
    get_stable_number(intval_x);
disp(sum(binary_num));
splits=2^(sum(binary_num));
Cons=cell(1,2^(sum(binary_num)));
output_cons_Aeq=cell(1,2^(sum(binary_num)));
output_cons_beq=cell(1,2^(sum(binary_num)));
Cons_index=0;
b1=binary_value{1};  % the unstable neurons in first hidden
b2=binary_value{2};  % the unstable neurons in second hidden
%% 
% all stable
% the derivative and equality constrain of barrier
if sum(binary_num)==0
    k=[1;-1];
    t31=W{3}*k;
    t32=b{3}*k;
    tb21=W{2}.*b2;
    tb22=b{2}.*b2;
    t21=tb21*t31;
    t22=tb22*t31+t32;
    tb11=W{1}.*b1;
    tb12=b{1}.*b1;
    t11=tb11*t21;
    t12=tb12*t21+t22;
    AAeq=t11';
    bbeq=-t12;
    output_cons_Aeq{1}=AAeq;
    output_cons_beq{1}=bbeq;
    r=((((([1,-1])*W{3}').*b2)*W{2}').*b1)*W{1}';
    splits=0; % 0 distinguish from 1 case of unstable neurons
    return;
end
%%
% there exist unstable neurons
bl1=binary_value_zero_loc{1};  % the location of unstable neurons in first hidden
bl2=binary_value_zero_loc{2};  % the location of unstable neurons in second hidden
bn1=binary_num(1);  % the number of unstable neurons in first hidden
bn2=binary_num(2);  % the number of unstable neurons in second hidden
% one case for a special unstable neurons setting
AA=zeros(sum(binary_num),2);
bb=zeros(sum(binary_num),1);
tCons=cell(1,3);
%%
for i=1:2^bn1  % the cases of first layer
    ci=dec2bin(i-1,bn1);
    for ibn=1:bn1  % set the unstable neurons value
        value_set=str2num(ci(ibn));  % setting neurons 0 or 1
        value_set_loc=bl1(ibn);  % the location of neurons
        b1(value_set_loc)=value_set;
        % A and b in this case
        tAA=zeros(1,input_size);
        if value_set==0  % inactive
            for tin=1:input_size
                tAA(tin)=W{1}(tin,value_set_loc);
            end
            AA(ibn,:)=tAA;
            bb(ibn)=-b{1}(value_set_loc);
        else
            for tin=1:input_size
                tAA(tin)=-W{1}(tin,value_set_loc);
            end
            AA(ibn,:)=tAA;
            bb(ibn)=b{1}(value_set_loc);
        end
    end
    %% 
    % check if this case is possible
    model.obj=[1,1];  % no matter, only check the constrains area
    model.A=sparse(AA(1:bn1,:));
    model.rhs=bb(1:bn1);
    sense=blanks(bn1);
    for sense_index=1:bn1
        sense(sense_index)='<';
    end
    model.sense=sense;
    model.lb=inf(intval_x);
    model.ub=sup(intval_x);
    model.vtype='CC';

    params.TimeLimit=100;
    params.OutputFlag=0;
    result = gurobi(model, params);
    count_lp=count_lp+1;
    time_lp=time_lp+result.runtime;
    if strcmp(result.status,'INFEASIBLE')  % this case is impossible
        continue;
    end
    %%
    for j=1:2^bn2  % the cases of first layer
        cj=dec2bin(j-1,bn2);
        for jbn=1:bn2  % set the unstable neurons value
            value_set=str2num(cj(jbn));
            value_set_loc=bl2(jbn);
            b2(value_set_loc)=value_set;
            % A and b in this case
            tAA=zeros(1,input_size);
            tbb=0;
            if value_set==0  % inactive, second hidder layer
                for k=1:size(b1,2)
                    if b1(k)==0  % inactive, first hidden, no contribution
                    else
                        for tin=1:input_size
                            tAA(tin)=tAA(tin)+W{2}(k,value_set_loc)*W{1}(tin,k);
                        end
                        tbb=tbb-W{2}(k,value_set_loc)*b{1}(k);
                    end
                end
                tbb=tbb-b{2}(value_set_loc);
                AA(bn1+jbn,:)=tAA;
                bb(bn1+jbn)=tbb;
            else
                for k=1:size(b1,2)
                    if b1(k)==0  % inactive, first hidden, no contribution
                    else
                        for tin=1:input_size
                            tAA(tin)=tAA(tin)-W{2}(k,value_set_loc)*W{1}(tin,k);
                        end
                        tbb=tbb+W{2}(k,value_set_loc)*b{1}(k);
                    end
                end
                tbb=tbb+b{2}(value_set_loc);
                AA(bn1+jbn,:)=tAA;
                bb(bn1+jbn)=tbb;
            end
        end
        %% 
        % the equality constrain of barrier
        k=[1;-1];
        t31=W{3}*k;
        t32=b{3}*k;
        tb21=W{2}.*b2;
        tb22=b{2}.*b2;
        t21=tb21*t31;
        t22=tb22*t31+t32;
        tb11=W{1}.*b1;
        tb12=b{1}.*b1;
        t11=tb11*t21;
        t12=tb12*t21+t22;
        AAeq=t11';
        bbeq=-t12;
        %% 
        % check if this case is possible
        model.obj=[1,1];  % no matter, only check the constrains area
        model.A=sparse([AA;AAeq]);
        model.rhs=[bb;bbeq];
        sense=blanks(bn1+bn2+1);
        for sense_index=1:bn1+bn2
            sense(sense_index)='<';
        end
        sense(bn1+bn2+1)='=';
        model.sense=sense;
        model.lb=inf(intval_x);
        model.ub=sup(intval_x);
        model.vtype='CC';

        params.TimeLimit=100;
        params.OutputFlag=0;
        result = gurobi(model, params);
        count_lp=count_lp+1;
        time_lp=time_lp+result.runtime;
        if strcmp(result.status,'INFEASIBLE')  % this case do not cross barrier
            continue;
        end

        %%
        % saving this case
        Cons_index=Cons_index+1;
        output_cons_Aeq{Cons_index}=AAeq;
        output_cons_beq{Cons_index}=bbeq;
        % derivative value
        r=((((([1,-1])*W{3}').*b2)*W{2}').*b1)*W{1}'; 
        tCons{1}=r;
        tCons{2}=AA;
        tCons{3}=bb;
        Cons{Cons_index}=tCons;
    end
end

Cons=Cons(1,1:Cons_index);
splits=Cons_index;
output_cons_Aeq=output_cons_Aeq(1,1:Cons_index);
output_cons_beq=output_cons_beq(1,1:Cons_index);
return;


%%
    % get stable and unstable numbers, recode the y before relu of every
    % layer£¬the binary location
    function [active_num,negative_num,binary_num,layer_outputs,variable_num,binary_value,binary_value_zero_loc]=...
            get_stable_number(y)
        t_num_layers=size(W,2);
        % the recode of hidden layer
        negative_num=zeros(1,t_num_layers-1);
        active_num=zeros(1,t_num_layers-1);
        binary_num=zeros(1,t_num_layers-1);
        variable_num=zeros(1,t_num_layers-1);
        
        layer_outputs=cell(1,t_num_layers);%before relu
        layer_outputs_after_relu=cell(1,t_num_layers);
        
        binary_value=cell(1,2);
        binary_value_t1=zeros(1,net_structure(3));
        binary_value_t2=zeros(1,net_structure(4));
        
        binary_value_zero_loc=cell(1,2);
        binary_value_zero_loc_t1=zeros(1,net_structure(3));
        binary_value_zero_loc_t2=zeros(1,net_structure(4));
        binary_value_zero_loc_index_t1=0;
        binary_value_zero_loc_index_t2=0;
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
                binary_value_t1(tj)=0;
            elseif inf(layer_outputs{t_layer_index}(tj))>=0
                t_layer_active_num=t_layer_active_num+1;
                layer_variable_num_length=layer_variable_num_length-2;
                binary_value_t1(tj)=1;
            else
                t_layer_binary_num=t_layer_binary_num+1;
                binary_value_zero_loc_index_t1=binary_value_zero_loc_index_t1+1;
                binary_value_zero_loc_t1(binary_value_zero_loc_index_t1)=tj;
                binary_value_t1(tj)=-1;
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
                        binary_value_t2(tj)=0;
                    elseif inf(layer_outputs{t_layer_index}(tj))>=0
                        t_layer_active_num=t_layer_active_num+1;
                        layer_variable_num_length=layer_variable_num_length-2;
                        binary_value_t2(tj)=1;
                    else
                        t_layer_binary_num=t_layer_binary_num+1;
                        binary_value_zero_loc_index_t2=binary_value_zero_loc_index_t2+1;
                        binary_value_zero_loc_t2(binary_value_zero_loc_index_t2)=tj;
                        binary_value_t2(tj)=-1;
                    end
                    layer_variable_num_length=layer_variable_num_length+3;
                end
                negative_num(t_layer_index)=t_layer_negative_num;
                active_num(t_layer_index)=t_layer_active_num;
                binary_num(t_layer_index)=t_layer_binary_num;
                variable_num(t_layer_index)=layer_variable_num_length;
            end
        end
        binary_value{1}=binary_value_t1;
        binary_value{2}=binary_value_t2;
        
        binary_value_zero_loc{1}=binary_value_zero_loc_t1;
        binary_value_zero_loc{2}=binary_value_zero_loc_t2;
    end

end
