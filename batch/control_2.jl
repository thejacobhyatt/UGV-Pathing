using JuMP, Gurobi, CSV, DataFrames

const GRB_ENV = Gurobi.Env()


scenario_name = "fixed"
w=51
h=51
batteryCapacity = 10000
τ = 60*60*60; #number of seconds to traverse the graph
s = 1 #the starting node of the troops
f = 2601 #the destination node of the troops
N=w*h*2;
A = 45602


dir2 = "batch/";

wh_string=string(w)*"_"*string(h)
arcs_file="arcs_"*wh_string*"_"*scenario_name*".csv";
tri_file=wh_string*"_triangle.csv"

arcs= CSV.read(arcs_file, DataFrame, header=0);
tri= CSV.read(tri_file, DataFrame, header=0);

ins = CSV.read(wh_string*"_inflow.csv", DataFrame, header=0);
outs = CSV.read(wh_string*"_outflow.csv", DataFrame, header=0);


D_lin = arcs[!,4];
T_lin = arcs[!,5];
E_lin = arcs[!,6];

tris=[]
for t in eachrow(tri)
    new_t=[]
    for a in t
        if a!=0
            append!(new_t,a)
        end
    end
    push!(tris,new_t)
end
IN=size(ins,1)
inflow=[]
outflow=[]
for i in 1:IN
    new_in=[]
    new_out=[]
    for j in 1:10
        if ins[i,j]!=0
            append!(new_in,ins[i,j])
            append!(new_out,outs[i,j])
        end
    end
    push!(inflow,new_in)
    push!(outflow,new_out)
end    



function two_step_en(w,h,s,f,A,N,d,t,E,tris,τ,inflow,outflow, alpha, beta)
    m = Model(() -> Gurobi.Optimizer(GRB_ENV));
    MAXTIME=120
    set_optimizer_attributes(m, "TimeLimit" => MAXTIME, "MIPGap" => 1e-2, "OutputFlag" => 0);
    @variable(m, x[1:A], Bin)
    @variable(m, batteryLevel[1:A] >= 0) # battery level at each step
    @objective(m, Min, alpha*sum(d[i]*x[i] for i in 1:A) + beta*(sum(E[i]*x[i] for i in 1:A)))
    ## (1/maximum(E))*
    @constraint(m, [i in 1:N], sum(x[k] for k in inflow[i])<=1)
    @constraint(m, [i in 1:N], sum(x[k] for k in outflow[i])<=1)
    @constraint(m, [i in 1:N], sum(x[k] for k in inflow[i] if i!=f && i!=s)-sum(x[k] for k in outflow[i] if i!=f && i!=s)==0)
    @constraint(m, sum(x[k] for k in inflow[s])-sum(x[k] for k in outflow[s])==-1)
    @constraint(m, sum(x[k] for k in inflow[f])-sum(x[k] for k in outflow[f])==1)
    @constraint(m, batteryLevel[1] == batteryCapacity) # initialize battery level

    tri_tot=length(tris)
    @constraint(m, [t in 1:tri_tot], sum(x[t] for t in tris[t])<=1)
    @constraint(m, sum(t[i]*x[i] for i=1:A) <= τ)
    @constraint(m, [i in 2:A], batteryLevel[i] == batteryLevel[i-1] - E[i-1] * x[i-1])  # update battery level at each step
    @constraint(m, [i in 1:A], batteryLevel[i] >= 0) # make sure it is never below 0
    @constraint(m, [i in 1:A], batteryLevel[i] <= batteryCapacity)
    @constraint(m, sum(E[i]*x[i] for i=1:A) <= batteryCapacity)

    optimize!(m)
    #time=MOI.get(m, MOI.SolveTime())
    #D_opt=objective_value(m)
    
    #@objective(m, Min, sum(E[i]*x[i] for i in 1:A))
    #@constraint(m, sum(d[i]*x[i] for i in 1:A)<=D_opt)
    #optimize!(m)
    optimalpath = value.(x)
    objective = objective_value(m)
    #time+=MOI.get(m, MOI.SolveTime())
    return optimalpath, objective, time
end

function write_path(opt_path, A, file_name)
    path=[]
    for i in 1:A
        if opt_path[i]==1
            append!(path,i)
        end
    end
    K=length(path)
    path_table=zeros(K,2)
    for i in 1:K
        path_table[i]=path[i]
    end
    path_output3 = DataFrame(path_table, :auto);
    CSV.write(file_name, path_output3); 
end

step = 0.005

@time begin
    for i in 0:step:.05
        alpha = i
        beta = 1  # This makes beta go in the opposite direction of alpha
            # Convert alpha and beta to strings and format the file name
            file_name = "output_batch_$(alpha)_$(beta).csv"
            
            # Call the function with the specified parameters
            pathrob, objT, sTime = two_step_en(w, h, s, f, A, N, D_lin, T_lin, E_lin, tris, τ, inflow, outflow, alpha, beta)
            write_path(pathrob, A, file_name)
            println("Finished with Alpha: $alpha, Beta: $beta")
    end
end
