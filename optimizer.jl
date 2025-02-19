using JuMP, Gurobi, CSV, DataFrames

const GRB_ENV = Gurobi.Env()

scenario_name = "3x3"
w=12
h=12
batteryCapacity = 10000

dir2 = "./";
tri= CSV.read("Buckner_tris_12_12.csv", DataFrame, header=0);
arcs= CSV.read("Buckner_arcs_12_12.csv", DataFrame, header=0);
ins = CSV.read("Buckner_ins_12_12.csv", DataFrame, header=0);
outs = CSV.read("Buckner_outs_12_12.csv", DataFrame, header=0);

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

function write_path(opt_path, A)
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
    CSV.write("output_"*scenario_name*".csv", path_output3); 
end

#User Input and Initial Calculations
time_total = 100000; #number of seconds to traverse the graph
# S_max=7
# C_max=5
s = 1 #the starting node of the troops
f = 144 #the destination node of the troops
N=w*h*2;
A = 2312


function two_step_optimization(w, h, s, f, A, N, d, t, E, tris, time_total, inflow, outflow, batteryCapacity)
    m = Model(() -> Gurobi.Optimizer(GRB_ENV))
    MAXTIME = 120
    set_optimizer_attributes(m, "TimeLimit" => MAXTIME, "MIPGap" => 1e-5, "OutputFlag" => 1)

    # Step 1: Minimize detection to find the least-detection path
    # @variable(m, x[1:A], Bin)
    @variable(m, x[1:(A+1)], Bin)  # Extend x to include the virtual arc


    @objective(m, Min, sum(d[i] * x[i] for i in 1:A))

    # Flow constraints
    @constraint(m, [i in 1:N], sum(x[k] for k in inflow[i]) <= 1)
    @constraint(m, [i in 1:N], sum(x[k] for k in outflow[i]) <= 1)
    @constraint(m, [i in 1:N], 
        sum(x[k] for k in inflow[i] if i != f && i != s) - 
        sum(x[k] for k in outflow[i] if i != f && i != s) == 0)

    @constraint(m, sum(x[k] for k in inflow[s]) - sum(x[k] for k in outflow[s]) == -1)
    @constraint(m, sum(x[k] for k in inflow[f]) - sum(x[k] for k in outflow[f]) == 1)

    # Triangle constraints
    tri_tot = length(tris)
    @constraint(m, [t in 1:tri_tot], sum(x[t] for t in tris[t]) <= 1) 
    
    # Time constraint
    @constraint(m, sum(t[i] * x[i] for i in 1:A) <= time_total)

    @variable(m, batteryLevel[1:A] >= 0)  # Battery level at each node
    
    # seperate constraint 
    # do not want to iterate from arcs that point back at 1
    # vice versa for the end node 
    @constraint(m, batteryLevel[s] == batteryCapacity) # add a ghost arc that starts the battery level 
    # Define a virtual arc index (A + 1) and add it to the model
    # fake_arc = A + 1  # An extra arc for battery initialization
    
    # Virtual arc constraint: Supplies batteryCapacity to start node
    # @constraint(m, batteryLevel[s] == batteryCapacity * x[fake_arc])
    # @constraint(m, x[fake_arc] == 1) # Ensure that this virtual arc is always chosen
    # push!(inflow[s], fake_arc)     # Modify inflow to account for the new virtual arc


    # for i in 1:A  # Iterate over arcs
    #    k = arcs[i, "Column2"]  # Get the starting node of arc i
    
    #    if k != s  # Ignore the start node
    #        @constraint(m, batteryLevel[i] == sum(batteryLevel[j] for j in inflow[k]) - E[i] * x[i])
    #    end
    # end
    
    
    # Ensure battery does not exceed maximum capacity
    @constraint(m, [i in 1:N], batteryLevel[i] <= batteryCapacity)
    
    # Ensure battery does not go below zero
    @constraint(m, [i in 1:N], batteryLevel[i] >= 0)
    
    # Optimize the model
    optimize!(m)

    # Extract the optimal path and final objective value
    optimal_path = value.(x)
    final_objective = objective_value(m)

    println("Final energy cost: ", final_objective)

    return optimal_path, final_objective
end

@time begin
    pathrob,sTime = two_step_optimization(w,h,s,f,A,N,D_lin,T_lin,E_lin,tris,time_total,inflow,outflow, batteryCapacity)
    end 

write_path(pathrob, A)