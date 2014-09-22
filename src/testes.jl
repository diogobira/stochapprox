###############################################################
### Some tests...
###############################################################

#
require("stoch_approx_multivariate.jl")

## Covariance Matrix
s11,s22 = 0.3^2, 0.25^2
s12 = 0.3*0.25*0.5
s21 = s12
sigma = [[s11 s12],[s21 s22]]
drift = [0.15 0.10]

## Creating a tree using the multivariate GBM
start_values = [100.0  80.0]
root = PathsTree(start_values,1)
params = {:drift=>drift, :sigma=>sigma, :dt=>21/252}
number_of_childs = push!([2 for i=1:24],0);
add_childs(root, number_of_childs, multivariate_gbm, params) 	

## Creating a copy of original tree and running the stochastic approximation method
new_root = deepcopy(root);
lots_of_petelecos(new_root, 1000, number_of_childs, multivariate_gbm, params, Euclidean());
get_all_paths(root)
get_all_paths(new_root)
get_all_probs(new_root)

M = Array[]
get_all_paths(root,M)

M0 = Array[]
get_all_paths(new_root,M0)
