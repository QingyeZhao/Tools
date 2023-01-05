function [ok,c1,c2]=interval_compute_output_area(x_min,x_max)
%%
% input parameters:
% x_min: the minimum of x
% x_max: the maximum of x
% output parameters:
% ok: this area is safe and do not cross the barrier
% c1: the minimum of net output
% c2: the maximum of net output
%%
    y=infsup(x_min,x_max);
    global W b
    % before output layer, there are ReLUs
    for t_layer_index=1:size(W,2)-1
        y=y*W{t_layer_index}+b{t_layer_index};
        for i=1:size(y,2)
            if y(i)<0
                y(i)=0;
            end
        end
    end
    % output layer, no ReLU
    t_layer_index=size(W,2);
    y=y*W{t_layer_index}+b{t_layer_index};
    r=y(1)-y(2);
    c1=inf(r);
    c2=sup(r);
    if c1>=0 || c2<=0
        % this area do not cross the barrier, it it safe
        ok=1;
    else
        ok=0;
    end
    
end