###############################################################
### Some tests...
###############################################################

## Covariance and Correlation Matrices
s11,s22 = 0.3^2, 0.30^2;
s12 = 0.3*0.30*-0.5;
s21 = s12;
sigma = [[s11 s12],[s21 s22]];
r11,r22 = 1,1;
r12 = -0.5; r21 = s12;
rho = [[r11 r12],[r21 r22]];
drift = [0.30 0.10];

## Creating a tree using the multivariate GBM
start_values = [100.0  100.0];
root = PathsTree(start_values,1);
params = {:drift=>drift, :sigma=>sigma, :rho=>rho, :dt=>3*21/252};
number_of_childs = push!([3 for i=1:4],0);
add_childs(root, number_of_childs, multivariate_gbm, params); 	

## Creating a copy of original tree and running the stochastic approximation method
new_root = deepcopy(root);
lots_of_petelecos(new_root, 100000, number_of_childs, multivariate_gbm, params, normalized_euclidean_distance);

## Testing the initialization method
strategy = [0.5 0.5; 0.5 0.5; 0.5 0.5; 0.5 0.5]
initialize_objectiveValues(new_root, strategy, 1)

## Sanity Check of the Objective Function on Basic Cases
strategy = [0.5 0.5; 0.5 0.5; 0.5 0.5; 0.5 0.5]
main_objectiveFun(new_root, strategy)

strategy = [1 0; 1 0; 1 0; 1 0]
main_objectiveFun(new_root, strategy)

strategy = [1 0; 1 0; 0 1; 0 1]
main_objectiveFun(new_root, strategy)

strategy = [0 1; 0 1; 0 1; 0 1]
main_objectiveFun(new_root, strategy)

#################################################
## Optimization!
#################################################

### reference example: https://github.com/JuliaOpt/NLopt.jl

using JuMP
using NLopt

# The solver
m = Model(solver=NLoptSolver(algorithm=:LN_COBYLA))

# The decision variables
@defVar(m, wA1);
@defVar(m, wB1);
@defVar(m, wA2);
@defVar(m, wB2);
@defVar(m, wA3);
@defVar(m, wB3);
@defVar(m, wA4);
@defVar(m, wB4);

# The objective function
function f(wA1,wB1,wA2,wB2,wA3,wB3,wA4,wB4) 
	strategy = [wA1 wB1; wA2 wB2; wA3 wB3; wA4 wB4]
	main_objectiveFun(new_root, strategy)
end

# The constraints
@setNLObjective(m, Max, f(wA1,wB1,wA2,wB2,wA3,wB3,wA4,wB4))
@addNLConstraint(m, wA1+wB1 == 1)
@addNLConstraint(m, wA2+wB2 == 1)
@addNLConstraint(m, wA3+wB3 == 1)
@addNLConstraint(m, wA4+wB4 == 1)

# The initial solution
setValue(wA1,0.5);
setValue(wB1,0.5);
setValue(wA2,0.5);
setValue(wB2,0.5);
setValue(wA3,0.5);
setValue(wB3,0.5);
setValue(wA4,0.5);
setValue(wB4,0.5);

# Solving the model!
status = solve(m)

####

# The solver
o = Opt(:LD_COBYLA,8)

# Objective function
function g(x::Vector, grad::Vector)
	strategy = [x[1] x[2]; x[3] x[4]; x[5] x[6]; x[7] x[8]]
	return main_objectiveFun(new_root, strategy)	
end

# Eq. Contraints Function
function eq_constraints(result::Vector, x::Vector, grad::Matrix)
	result[1] = x[1]+x[2]-1
	result[2] = x[3]+x[4]-1		
	result[3] = x[5]+x[6]-1	
	result[4] = x[7]+x[8]-1	
end

max_objective!(o, g)
equality_constraint!(o, eq_constraints, [1.0e-6 for i=1:4]')
lower_bounds!(o, [0, 0, 0, 0, 0, 0, 0, 0])
upper_bounds!(o, [1, 1, 1, 1, 1, 1, 1, 1])
maxeval!(o, 500)

(optf,optx,ret) = optimize(o, [0.5 for i=1:8])


