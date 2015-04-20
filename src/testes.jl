###############################################################
### Some tests...
###############################################################

#
require("stoch_approx_multivariate.jl")

## Covariance Matrix
s11,s22 = 0.3^2, 0.30^2;
s12 = 0.3*0.30*0.5;
s21 = s12;
sigma = [[s11 s12],[s21 s22]];
r11,r22 = 1,1;
r12 = 0.5; r21 = s12;
rho = [[r11 r12],[r21 r22]];
drift = [0.15 0.10];

## Creating a tree using the multivariate GBM
start_values = [10000.0  10000.0];
root = PathsTree(start_values,1);
params = {:drift=>drift, :sigma=>sigma, :rho=>rho, :dt=>3*21/252};
number_of_childs = push!([3 for i=1:4],0);
add_childs(root, number_of_childs, multivariate_gbm, params); 	

## Creating a copy of original tree and running the stochastic approximation method
new_root = deepcopy(root);
lots_of_petelecos(new_root, 300000, number_of_childs, multivariate_gbm, params, normalized_euclidean_distance);

M = Array[]
get_all_paths(root,M)
write_multivariate_paths_to_file("original_paths.csv", M)

M0 = Array[]
get_all_paths(new_root,M0)
write_multivariate_paths_to_file("new_paths.csv", M0)

get_all_probs(new_root,"new_paths_probs.csv")
