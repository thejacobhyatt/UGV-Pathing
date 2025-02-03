using JuMP, Gurobi, CSV, DataFrames

const GRB_ENV = Gurobi.Env()

scenario_name = "3x3"
w=12
h=12
batteryCapacity = 100000

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
τ = 60*60*60; #number of seconds to traverse the graph
# S_max=7
# C_max=5
s = 1 #the starting node of the troops
f = 144 #the destination node of the troops
N=w*h*2;
A = 2312


function two_step_optimization(w, h, s, f, A, N, d, t, E, tris, τ, inflow, outflow, batteryCapacity)
    m1 = Model(() -> Gurobi.Optimizer(GRB_ENV))
    MAXTIME = 120
    set_optimizer_attributes(m1, "TimeLimit" => MAXTIME, "MIPGap" => 1e-2, "OutputFlag" => 1)

    # Step 1: Minimize detection to find the least-detection path
    @variable(m1, x[1:A], Bin)
    @objective(m1, Min, sum(d[i] * x[i] for i in 1:A))

    @constraint(m1, [i in 1:N], sum(x[k] for k in inflow[i]) <= 1)
    @constraint(m1, [i in 1:N], sum(x[k] for k in outflow[i]) <= 1)
    @constraint(m1, [i in 1:N], 
        sum(x[k] for k in inflow[i] if i != f && i != s) - 
        sum(x[k] for k in outflow[i] if i != f && i != s) == 0)
    @constraint(m1, sum(x[k] for k in inflow[s]) - sum(x[k] for k in outflow[s]) == -1)
    @constraint(m1, sum(x[k] for k in inflow[f]) - sum(x[k] for k in outflow[f]) == 1)

    tri_tot = length(tris)
    @constraint(m1, [t in 1:tri_tot], sum(x[t] for t in tris[t]) <= 1) # triangles
    @constraint(m1, sum(t[i] * x[i] for i in 1:A) <= τ) # time


    # Battery and energy tracking
    # @variable(m1, batteryLevel[1:A] >= 0) # Battery level at each step
    # @variable(m1, energyUsed[1:A] >= 0) # Energy used in each path transition
    
    # Initialize battery level
    # @constraint(m1, batteryLevel[1] == batteryCapacity) 

    # Energy constraints: energy consumed per path transition
   #  @constraint(m1, [i in 1:A], energyUsed[i] == E[i] * x[i])  # Energy consumption for each path

    # Update battery level for each step
    # @constraint(m1, [i in 2:A], batteryLevel[i] == batteryLevel[i-1] - energyUsed[i])  # For i > 1

    # @constraint(m1, [i in 1:A], batteryLevel[i] >= 0) # Battery does not go below 0
    # @constraint(m1, [i in 1:A], batteryLevel[i] <= batteryCapacity) # Ensure battery level does not exceed capacity

    # Insert code to handle jumps to top grid if needed (if there's logic for this)
    # For example, if battery is low, some paths could be avoided.

    optimize!(m1)

    optimal_path = value.(x)
    final_objective = objective_value(m1)

    println("Final energy cost: ", final_objective)

    return optimal_path, final_objective
end



@time begin
    pathrob,sTime = two_step_optimization(w,h,s,f,A,N,D_lin,T_lin,E_lin,tris,τ,inflow,outflow, batteryCapacity)
    end 

write_path(pathrob, A)