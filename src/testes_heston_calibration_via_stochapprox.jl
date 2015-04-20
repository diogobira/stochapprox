###################################################################
# Some Tests
###################################################################

require("stoch_approx_multivariate.jl")
require("heston_calibration_via_stochapprox.jl")
require("stock_options_via_stochapprox.jl")

## Creating a tree using the Heston Model
#start_values = [0.25^2 100.0];
#root = PathsTree(start_values,1);
#rho = [[1 -0.4],[-0.4 1]];
#params_heston = {:mu=>0.10, :v0=>0.35^2, :kappa=>63/252, :theta=>0.45^2, :sigma=>0.3, :rho=>rho, :dt=>21/252}
#params_heston = {:mu=>0.10, :v0=>0.35^2, :kappa=>63/252, :theta=>0.45^2, :sigma=>0.3, :rho=>rho, :dt=>2*21/252}
#number_of_childs = push!([2 for i=1:12],0);
#number_of_childs = push!([3 for i=1:6],0);
#number_of_childs = [2 3 2 3 2 3 2 3 2 3 2 3 0];
#number_of_childs = [5 5 5 5 5 5 0];
#add_childs(root, number_of_childs, heston_path, params_heston); 	
#new_root = deepcopy(root);
#lots_of_petelecos(new_root, 100000, number_of_childs, heston_path, params_heston, euclidean_dist);

## Creating a tree using the GBM Model
start_values = [100.0 100.0];
root = PathsTree(start_values,1);
s11,s22 = 0.40^2, 0.30^2;
s12 = 0.40*0.30*-0.5;
s21 = s12;
sigma = [[s11 s12],[s21 s22]];
r11,r22 = 1,1;
r12 = -0.5; r21 = s12;
rho = [[r11 r12],[r21 r22]];
drift = [0.10 0.10];
params_gbm = {:drift=>drift, :sigma=>sigma, :rho=>rho, :dt=>21/252};
number_of_childs = push!([2 for i=1:12],0);
#number_of_childs = push!([2 for i=1:6],0);
#number_of_childs = [2 3 2 3 2 3 2 3 2 3 2 3 0];
#number_of_childs = [2 2 2 2 1 1 1 1 1 10 10 10 0];
add_childs(root, number_of_childs, multivariate_gbm, params_gbm); 	
new_root = deepcopy(root);
#lots_of_petelecos(new_root, 50000, number_of_childs, multivariate_gbm, params_gbm, euclidean_dist);
lots_of_petelecos(new_root, 50000, number_of_childs, multivariate_gbm, params_gbm, normalized_euclidean_distance);
lots_of_petelecos(new_root, 100000, number_of_childs, multivariate_gbm, params_gbm, normalized_euclidean_distance);

## Creating a tree using the Heston Model
#start_values = [0.25^2 100.0];
#root = PathsTree(start_values,1);
#rho = [[1 -0.4],[-0.4 1]];
#params_heston = {:mu=>0.10, :v0=>0.35^2, :kappa=>63/252, :theta=>0.45^2, :sigma=>0.3, :rho=>rho, :dt=>21/252}
#params_heston = {:mu=>0.10, :v0=>0.35^2, :kappa=>63/252, :theta=>0.45^2, :sigma=>0.3, :rho=>rho, :dt=>3*21/252}
#number_of_childs = push!([2 for i=1:12],0);
#number_of_childs = push!([5 for i=1:4],0);
#number_of_childs = [2 3 2 3 2 3 2 3 2 3 2 3 0];
#number_of_childs = [5 5 5 5 5 5 0];
#add_childs(root, number_of_childs, heston_path, params_heston); 	
#new_root = deepcopy(root);
#lots_of_petelecos(new_root, 300000, number_of_childs, heston_path, params_heston, euclidean_dist);

#leafs = PathsTree[]
#get_leafs(new_root,leafs)
#[(x.value[2],x.prob) for x=leafs]
#[(x.optionValue,x.prob) for x=leafs]

### Saving original tree to file
M0 = []
get_all_paths(new_root, M0) 
write_multivariate_paths_to_file("original.gbm.path.csv", M0)
#write_multivariate_paths_to_file("original.heston.path.csv", M0)

### Saving rebalanced tree to file
M = []
get_all_paths(new_root, M) 
write_multivariate_paths_to_file("gbm.path.csv", M)
#write_multivariate_paths_to_file("heston.path.csv", M)

### Pricing an European Option using the original and the rebalanced tree
step=1
K=110
discountFactor = exp(-0.1*step*21/252)
price_european_call(new_root, K, discountFactor, 2)



