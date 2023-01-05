%%
% INTLAB: http://www.ti3.tu-harburg.de/rump/intlab/
% Gurobi: https://www.gurobi.com
% 
%%
intvalinit('DisplayInfsup');
clear global;
clear;
clc;
close all;
fclose('all'); 
warning('off');
%%
tic;
cputime0=cputime;
save_area_cons=0;  % save derivative area and constrains for draw
%% 
% the number and time of milp, lp, and ploy_p
global count_milp count_lp count_gqp
global time_milp time_lp time_gqp

count_milp=2; 
count_lp=0;
count_gqp=0;

time_milp=0; 
time_lp=0; 
time_gqp=0; 

%%
% get network parameters
global net_structure
net_structure=load('net/structure');
global W b
for i=1:net_structure(1)-1
    W{i}=load(['net/w',num2str(i)]);
    b{i}=load(['net/b',num2str(i)])';
end
%%
% verify unsafe set
unsafe_min=[-2,-2];
unsafe_max=[-1,-1];
[~,~,c2]=milp_output_area(unsafe_min,unsafe_max);
if c2>0
    error('Error net for unsafe region!\n');
end
%%
% verify initial set
initial_min=[-0.2,0.3];
initial_max=[0.2,0.7];
[~,c1,~]=milp_output_area(initial_min,initial_max);
if c1<0
    error('Error net for initial set!\n');
end
%%
% set the invariant area
invariant_min=[-2,-2];
invariant_max=[2,2];

%%
% split the invariant area to small pieces by dimision
% polynomial can not be neg, so split the invariant to double pieces
% do not cross zero
piece=[10,10];

% the step length of every dimision
step=(invariant_max-invariant_min)./piece;

%%
% the output error area file
log_dir='./';
fid_error_area=fopen([log_dir,'error_area.txt'],'w');

%% 
% begin verify
x_min=invariant_min;
x_max=invariant_max;
for i=1:piece(1)
    % the small piece
    x_min(1)=invariant_min(1)+(i-1)*step(1);
    x_max(1)=invariant_min(1)+(i)*step(1);
    for j=1:piece(2)
        % the small piece
        x_min(2)=invariant_min(2)+(j-1)*step(2);
        x_max(2)=invariant_min(2)+(j)*step(2);
        
        % interval compute the output area
        [flag_ok,c1,c2]=interval_compute_output_area(x_min,x_max);
        if flag_ok  % this area do not cross barrier
            continue;
        end
        
        % get output area through milp, this is a accurate computation
        [flag_ok,c1,c2]=milp_output_area(x_min,x_max);
        if flag_ok  % this area do not cross barrier
            continue;
        end
        
        % this area cross barrier, we need verify the derivative >= 0
        [flag_ok,most_min,most_min_x]=derivative_verify_groubi(x_min,x_max,save_area_cons);
        if flag_ok  % the derivative is always >= 0
            continue;
        else
            for te=1:size(x_min,2)
                fprintf(fid_error_area,'%f\t',x_min(te));
            end
            for te=1:size(x_max,2)
                fprintf(fid_error_area,'%f\t',x_max(te));
            end
            fprintf(fid_error_area,'\n');
        end
    end
end
fclose(fid_error_area);
cputime1=cputime;

%%
ps=1;
for i=1:size(piece,2)
    ps=ps*piece(i);
end
fprintf(['sum area splits:  ',num2str(ps),'\n']);
fprintf(['count_milp:  ',num2str(count_milp),'\t\ttime_milp:  ',num2str(time_milp),'\n']);
fprintf(['count_lp:  ',num2str(count_lp),'\t\ttime_lp:  ',num2str(time_lp),'\n']);
fprintf(['count_gqp:  ',num2str(count_gqp),'\t\ttime_gqp:  ',num2str(time_gqp),'\n']);
fprintf(['sum programming time:  ',num2str(time_milp+time_lp+time_gqp),'\n']);
fprintf(['cpu time:  ',num2str(cputime1-cputime0),'\n']);

fid_log=fopen([log_dir,'log_',num2str(ps),'_pieces.txt'],'w');
fprintf(fid_log,'%s\t%d\n','sum area splits:',ps);
fprintf(fid_log,'%s\t%d\t%s\t%f\n','count_milp:',count_milp,'time_milp:',time_milp);
fprintf(fid_log,'%s\t%d\t%s\t%f\n','count_lp:',count_lp,'time_lp:',time_lp);
fprintf(fid_log,'%s\t%d\t%s\t%f\n','count_gqp:',count_gqp,'time_gqp:',time_gqp);
fprintf(fid_log,'%s\t%f\n','sum programming time:',time_milp+time_lp+time_gqp);
fprintf(fid_log,'%s\t%f\n','cpu time:',cputime1-cputime0);
fclose(fid_log);

toc;





