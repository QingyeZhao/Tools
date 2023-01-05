function [ok,most_min,most_min_x]=derivative_verify_groubi(x_min,x_max,save_area_cons)
%%
% input parameters:
% x_min: the minimum of x
% x_max: the maximum of x
% output parameters:
% ok: this area is safe and do not cross the barrier
% c1: the minimum of net output
% c2: the maximum of net output
%%
% polynomial can not be neg, so split the invariant to double pieces
% if x1>=0 and x2>=0
% x1'=-x1+2*x1^3*x2^2
% x2'=-x2
% minimize 2*d1*x3*x4+(-d1*x1)+(-d2*x2)
%          x3=x1^3
%          x4=x2^2
%%
% if x1>=0 and x2<=0
% minimize 2*d1*x3*x4+(-d1*x1)+(d2*x2)
%          x3=x1^3
%          x4=x2^2
%
%%
% if x1<=0 and x2>=0
% minimize -2*d1*x3*x4+(d1*x1)+(-d2*x2)
%          x3=x1^3
%          x4=x2^2
%
%%
% if x1<=0 and x2<=0
% minimize -2*d1*x3*x4+(d1*x1)+(d2*x2)
%          x3=x1^3
%          x4=x2^2
%

%%
global W b
global count_gqp
global time_gqp
ok=0;
most_min=Inf;
most_min_x=zeros(1,2);
%%

vlb=x_min;
vub=x_max;
[r,Cons,splits,output_cons_Aeq,output_cons_beq]=get_derivative_constrain(x_min,x_max);
if splits==0
    count_gqp=count_gqp+1;
    if save_area_cons
        save_deri_area_big(num2str(count_gqp),vlb,vub);
    end
    
    %%
    model.rhs=output_cons_beq{1};
    model.sense='=';
    model.vtype='CCCC';
    % x3=x1^3
    model.genconpow(1).xvar = 1;
    model.genconpow(1).yvar = 3;
    model.genconpow(1).a = 3;
    % x4=x2^2
    model.genconpow(2).xvar = 2;
    model.genconpow(2).yvar = 4;
    model.genconpow(2).a = 2;
    
%     params.TimeLimit=100;
%     params.OutputFlag=0;
    params.FuncPieces = 1;
    params.FuncPieceLength = 1e-4;
    params.NonConvex=2;
    params.OutputFlag=0;
    %%
    if vlb(1)>=0 && vlb(2)>=0
        % x1>=0 and x2>=0
        model.Q=sparse([
            [0,0,0,0];
            [0,0,0,0];
            [0,0,0,2*r(1)];
            [0,0,0,0]
            ]);
        model.obj=[-r(1),-r(2),0,0];
        model.lb=[vlb(1),vlb(2),vlb(1)*vlb(1)*vlb(1),vlb(2)*vlb(2)];
        model.ub=[vub(1),vub(2),vub(1)*vub(1)*vub(1),vub(2)*vub(2)];
        tAeq=output_cons_Aeq{1};
        model.A=sparse([tAeq,0,0]);
        result = gurobi(model, params);
        most_min=result.objval;
        most_min_x=result.x(1:2);
        time_gqp=time_gqp+result.runtime;

    elseif vlb(1)>=0 && vub(2)<=0
        %%
        % x1>=0 and x2<0
        model.Q=sparse([
            [0,0,0,0];
            [0,0,0,0];
            [0,0,0,2*r(1)];
            [0,0,0,0]
            ]);
        model.obj=[-r(1),r(2),0,0];
        model.lb=[vlb(1),-vub(2),vlb(1)*vlb(1)*vlb(1),(-vub(2))*(-vub(2))];
        model.ub=[vub(1),-vlb(2),vub(1)*vub(1)*vub(1),(-vlb(2))*(-vlb(2))];
        tAeq=output_cons_Aeq{1};
        tAeq(2)=-tAeq(2);
        model.A=sparse([tAeq,0,0]);
        result = gurobi(model, params);
        most_min=result.objval;
        most_min_x=result.x(1:2);
        most_min_x(2)=-most_min_x(2);
        time_gqp=time_gqp+result.runtime;

    elseif vub(1)<=0 && vlb(2)>=0
        %%
        % x1<0 and x2>=0
        model.Q=sparse([
            [0,0,0,0];
            [0,0,0,0];
            [0,0,0,-2*r(1)];
            [0,0,0,0]
            ]);
        model.obj=[r(1),-r(2),0,0];
        model.lb=[-vub(1),vlb(2),(-vub(1))*(-vub(1))*(-vub(1)),(vlb(2))*(vlb(2))];
        model.ub=[-vlb(1),vub(2),(-vlb(1))*(-vlb(1))*(-vlb(1)),(vub(2))*(vub(2))];
        tAeq=output_cons_Aeq{1};
        tAeq(1)=-tAeq(1);
        model.A=sparse([tAeq,0,0]);
        result = gurobi(model, params);
        most_min=result.objval;
        most_min_x=result.x(1:2);
        most_min_x(1)=-most_min_x(1);
        time_gqp=time_gqp+result.runtime;
    
    elseif vub(1)<=0 && vub(2)<=0
        %%
        % x1<0 and x2<0
        model.Q=sparse([
            [0,0,0,0];
            [0,0,0,0];
            [0,0,0,-2*r(1)];
            [0,0,0,0]
            ]);
        model.obj=[r(1),r(2),0,0];
        model.lb=[-vub(1),-vub(2),(-vub(1))*(-vub(1))*(-vub(1)),(-vub(2))*(-vub(2))];
        model.ub=[-vlb(1),-vlb(2),(-vlb(1))*(-vlb(1))*(-vlb(1)),(-vlb(2))*(-vlb(2))];
        tAeq=output_cons_Aeq{1};
        tAeq(1)=-tAeq(1);
        tAeq(2)=-tAeq(2);
        model.A=sparse([tAeq,0,0]);
        result = gurobi(model, params);
        most_min=result.objval;
        most_min_x=result.x(1:2);
        most_min_x(1)=-most_min_x(1);
        most_min_x(2)=-most_min_x(2);
        time_gqp=time_gqp+result.runtime;

    else
        error('Bad splits!\n');
    end
    %%
    if most_min>=0
        ok=1;
    else
        ok=0;
        fprintf(['input:  ',num2str(most_min_x),...
            '\t\toutput:  ',num2str(get_net_output(fval_x)),...
            '\t\tderivative:  ',num2str(most_min),'\n']);
    end
    fprintf([num2str(most_min),'\n']);
else
    for i=1:splits
        all=Cons{i};
        r=all{1};
        A_p=all{2};
        b_p=all{3};
        Aeq_p=output_cons_Aeq{i};
        beq_p=output_cons_beq{i};
        tA_Aeq=[A_p;Aeq_p];
        tA_Aeq=[tA_Aeq,zeros(size(tA_Aeq,1),2)];
        count_gqp=count_gqp+1;
        if save_area_cons
            save_deri_area_small(num2str(count_gqp),vlb,vub,A_p,b_p);
        end
        %%
        model.rhs=[b_p;beq_p];
        tsense=blanks(size(b_p,1)+1);
        for i_tsense=1:size(b_p,1)
            tsense(i_tsense)='<';
        end
        tsense(size(b_p,1)+1)='=';
        model.sense=tsense;
        model.vtype='CCCC';
        % x3=x1^3
        model.genconpow(1).xvar = 1;
        model.genconpow(1).yvar = 3;
        model.genconpow(1).a = 3;
        % x4=x2^2
        model.genconpow(2).xvar = 2;
        model.genconpow(2).yvar = 4;
        model.genconpow(2).a = 2;

    %     params.TimeLimit=100;
    %     params.OutputFlag=0;
        params.FuncPieces = 1;
        params.FuncPieceLength = 1e-4;
        params.NonConvex=2;
        params.OutputFlag=0;
        
        %%
        if vlb(1)>=0 && vlb(2)>=0
            % x1>=0 and x2>=0
            model.Q=sparse([
                [0,0,0,0];
                [0,0,0,0];
                [0,0,0,2*r(1)];
                [0,0,0,0]
                ]);
            model.obj=[-r(1),-r(2),0,0];
            model.lb=[vlb(1),vlb(2),vlb(1)*vlb(1)*vlb(1),vlb(2)*vlb(2)];
            model.ub=[vub(1),vub(2),vub(1)*vub(1)*vub(1),vub(2)*vub(2)];
            model.A=sparse(tA_Aeq);
            result = gurobi(model, params);
            most_min=result.objval;
            most_min_x=result.x(1:2);
            time_gqp=time_gqp+result.runtime;
            
        elseif vlb(1)>=0 && vub(2)<=0
            %%
            % x1>=0 and x2<0
            model.Q=sparse([
                [0,0,0,0];
                [0,0,0,0];
                [0,0,0,2*r(1)];
                [0,0,0,0]
                ]);
            model.obj=[-r(1),r(2),0,0];
            model.lb=[vlb(1),-vub(2),vlb(1)*vlb(1)*vlb(1),(-vub(2))*(-vub(2))];
            model.ub=[vub(1),-vlb(2),vub(1)*vub(1)*vub(1),(-vlb(2))*(-vlb(2))];
            tA_Aeq(:,2)=-tA_Aeq(:,2);
            model.A=sparse(tA_Aeq);
            result = gurobi(model, params);
            most_min=result.objval;
            most_min_x=result.x(1:2);
            most_min_x(2)=-most_min_x(2);
            time_gqp=time_gqp+result.runtime;
        elseif vub(1)<=0 && vlb(2)>=0
            %%
            % x1<0 and x2>=0
            model.Q=sparse([
                [0,0,0,0];
                [0,0,0,0];
                [0,0,0,-2*r(1)];
                [0,0,0,0]
                ]);
            model.obj=[r(1),-r(2),0,0];
            model.lb=[-vub(1),vlb(2),(-vub(1))*(-vub(1))*(-vub(1)),(vlb(2))*(vlb(2))];
            model.ub=[-vlb(1),vub(2),(-vlb(1))*(-vlb(1))*(-vlb(1)),(vub(2))*(vub(2))];
            tA_Aeq(:,1)=-tA_Aeq(:,1);
            model.A=sparse(tA_Aeq);
            result = gurobi(model, params);
            most_min=result.objval;
            most_min_x=result.x(1:2);
            most_min_x(1)=-most_min_x(1);
            time_gqp=time_gqp+result.runtime;

        elseif vub(1)<=0 && vub(2)<=0
            %%
            % x1<0 and x2<0
            model.Q=sparse([
                [0,0,0,0];
                [0,0,0,0];
                [0,0,0,-2*r(1)];
                [0,0,0,0]
                ]);
            model.obj=[r(1),r(2),0,0];
            model.lb=[-vub(1),-vub(2),(-vub(1))*(-vub(1))*(-vub(1)),(-vub(2))*(-vub(2))];
            model.ub=[-vlb(1),-vlb(2),(-vlb(1))*(-vlb(1))*(-vlb(1)),(-vlb(2))*(-vlb(2))];
            tA_Aeq(:,1)=-tA_Aeq(:,1);
            tA_Aeq(:,2)=-tA_Aeq(:,2);
            model.A=sparse(tA_Aeq);
            result = gurobi(model, params);
            most_min=result.objval;
            most_min_x=result.x(1:2);
            most_min_x(1)=-most_min_x(1);
            most_min_x(2)=-most_min_x(2);
            time_gqp=time_gqp+result.runtime;
        else
            error('Bad splits!\n');
        end
        %%
        if most_min>=0
            ok=1;
        else
            ok=0;
            fprintf(['input:  ',num2str(most_min_x),...
                '\t\toutput:  ',num2str(get_net_output(fval_x)),...
                '\t\tderivative:  ',num2str(most_min),'\n']);
        end
        fprintf([num2str(most_min),'\n']);
        
        
    end
end



%%
    function [ll,uu]=get_tight_bount(tA,tb,tAeq,tbeq,tlb,tub)
        modelt.A=sparse([tA;tAeq]);
        modelt.rhs=[tb;tbeq];
        sense=blanks(size(tA,2)+1);
        for sense_index=1:size(tA,2)
            sense(sense_index)='<';
        end
        sense(size(tA,2)+1)='=';
        modelt.sense=sense;
        modelt.lb=tlb;
        modelt.ub=tub;
        modelt.vtype='CC';

        paramst.TimeLimit=100;
        paramst.OutputFlag=0;
        
        ll=tlb;
        uu=tub;
        
        for it=1:size(W{1},1)
            tobj=zeros(1,size(W{1},1));
            tobj(it)=1;
            modelt.obj=tobj;
            resultt = gurobi(modelt, paramst);
            ll(it)=resultt.objval;
        end
        modelt.modelsense = 'max';
        for it=1:size(W{1},1)
            tobj=zeros(1,size(W{1},1));
            tobj(it)=1;
            modelt.obj=tobj;
            resultt = gurobi(modelt, paramst);
            uu(it)=resultt.objval;
        end
    end
    function output=get_net_output(input)
        y=input;
        
        % before output layer
        for t_layer_index=1:size(W,2)-1
            y=y*W{t_layer_index}+b{t_layer_index};
            for ii=1:size(y,2)
                if y(ii)<0
                    y(ii)=0;
                end
            end
        end
        % output layer
        y=y*W{size(W,2)}+b{size(W,2)};
        output=y(1)-y(2);
    end
    function save_deri_area_big(name,mini,maxi)
        dir_name=['deri_area/Big/',name,'/'];
        if exist(dir_name,'dir')==0
            mkdir(dir_name);
        end
        fid_m=fopen([dir_name,'min.txt'],'w');
        for tm=1:size(mini,2)
            fprintf(fid_m,'%f\n',mini(tm));
        end
        fclose(fid_m);
        fid_m=fopen([dir_name,'max.txt'],'w');
        for tm=1:size(maxi,2)
            fprintf(fid_m,'%f\n',maxi(tm));
        end
        fclose(fid_m);
    end

    function save_deri_area_small(name,mini,maxi,sA,sb)
        dir_name=['deri_area/Small/',name,'/'];
        if exist(dir_name,'dir')==0
            mkdir(dir_name);
        end
        fid_m=fopen([dir_name,'min.txt'],'w');
        for tm=1:size(mini,2)
            fprintf(fid_m,'%f\n',mini(tm));
        end
        fclose(fid_m);
        fid_m=fopen([dir_name,'max.txt'],'w');
        for tm=1:size(maxi,2)
            fprintf(fid_m,'%f\n',maxi(tm));
        end
        fclose(fid_m);
        
        fid_m=fopen([dir_name,'sA.txt'],'w');
        for tm=1:size(sA,1)
            for tn=1:size(sA,2)
                fprintf(fid_m,'%f\t',sA(tm,tn));
            end
            fprintf(fid_m,'\n');
        end
        fclose(fid_m);
        fid_m=fopen([dir_name,'sb.txt'],'w');
        for tm=1:size(sb,1)
            for tn=1:size(sb,2)
                fprintf(fid_m,'%f\t',sb(tm,tn));
            end
            fprintf(fid_m,'\n');
        end
        fclose(fid_m);
    end

end
