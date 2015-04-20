###################################################################
# Some Tests
###################################################################

require("stoch_approx_multivariate.jl")
require("heston_calibration_via_stochapprox.jl")
require("stock_options_via_stochapprox.jl")

## Creating a tree using the GBM Model
start_values = [100.0 100.0];
root = PathsTree(start_values,1);
s11,s22 = 0.40^2, 0.30^2;
s12 = 0.40*0.30*-0.5;
s21 = s12;
sigma = [[s11 s12],[s21 s22]];
r11,r22 = 1,1;
#r12 = -0.5; r21 = s12;
r12 = 0.5; r21 = s12;
rho = [[r11 r12],[r21 r22]];
drift = [0.10 0.10];
params_gbm = {:drift=>drift, :sigma=>sigma, :rho=>rho, :dt=>21/252};

number_of_childs = push!([2 for i=1:12],0);
add_childs(root, number_of_childs, multivariate_gbm, params_gbm); 	

###
new_root = deepcopy(root);
lots_of_petelecos(new_root, 100000, number_of_childs, multivariate_gbm, params_gbm, normalized_euclidean_distance);

###
#new_root_2 = deepcopy(root);
#lots_of_petelecos_by_level(new_root_2, 100, multivariate_gbm, params_gbm, normalized_euclidean_distance);
#lots_of_petelecos(new_root_2, 100000, number_of_childs, multivariate_gbm, params_gbm, normalized_euclidean_distance);

### Pricing an European Option using the original and the rebalanced tree
step=1
K=110
discountFactor = exp(-0.1*step*21/252)
#price_european_call(new_root, K, discountFactor, 2)
price_european_call(new_root_2, K, discountFactor, 2)



