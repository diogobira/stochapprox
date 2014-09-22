###################################################################
# Stochastic Approximation 
# 
###################################################################

############################################
### Dependencies
############################################
using Distances

############################################
### Data Structures
############################################

### PathsTree Data Structure
type PathsTree

	## Attributes
	value
	level::Int32
	childs::Array{PathsTree,1}
	father::PathsTree
	counter::Int64
	prob::Float64
	
	## Constructors
	
	# Only for the root node 
	function PathsTree(value,level) 
		x = new() 
		x.value = value
		x.level = level
		x.childs = []
		x.counter = 0
		x.prob = 1
		x
	end
	
	# For generic nodes
	function PathsTree(value,level,father::PathsTree) 
		x = new() 
		x.value = value
		x.level = level
		x.childs = []
		x.father = father
		x.counter = 0
		x.prob = 0
		x
	end
	
end

############################################
### Auxiliary Functions
############################################

## multivariate_gbm
## Function to build a multivariate GBM path starting from a node (that contains the current value),
## given a multivariate drift, a covariance matrix, and a step size.
function multivariate_gbm(node, N, params)

	#Separate the params
	drift = params[:drift] #drift
	sigma = params[:sigma] #covariance matrix
	dt = params[:dt] #step size

	#Get the dimension off the process
	d = size(node.value)[2]

	#Create the correlated "shocks"
	A = chol(sigma)
	shocks = (A * randn(d,N))'

	#Volatilities
	vol = sqrt(diag(sigma))'

	#Array for the paths (different dimensions by column)
	path = zeros(N+1,d)
	path[1,:] = node.value
	for i=2:N+1
		path[i,:] = path[i-1,:] .* exp((drift .- 0.5*vol.^2 ) * dt +  sqrt(dt) * vol .*  shocks[i-1,:])
	end
	
	#If N==1, return only the next step values. Otherwise, returns the complete 
	#path including the initial values
	N==1 ? path[2,:] : path

end

###############################################################
### Some distance functions
###############################################################


###############################################################
### Auxiliary Tree Creation/Manipulation/Search Functions
###############################################################

### add_childs
### Function to build a tree a given specific function/params and the given number of nodes at each level
function add_childs(node, number_of_childs, path_function::Function, path_function_params)
	for i=1:number_of_childs[node.level]
		value = path_function(node, 1, path_function_params)
		child_node = PathsTree(value, node.level+1, node)
		push!(node.childs, child_node)
		add_childs(child_node, number_of_childs, path_function, path_function_params)
	end	
end

## get_nearest_child
## Function to get the index of child with (euclidean) minimum (normalized) 
## distance to the father. 
function get_nearest_child(node, value, dist)
	distances = [evaluate(dist, vec(n.value), vec(value)) for n = node.childs]
	min_distance = minimum(distances)
	idx = findfirst(distances, minimum(distances))
	return idx, min_distance 
end

## print
## Function to print the tree
import Base.print
function print(p::PathsTree)
	println(repeat(";",p.level-1),p.value)
	for i=1:length(p.childs)
		print(p.childs[i])
	end
end

## get_leafs
## Get all the leaf nodes of a tree
function get_leafs(node::PathsTree, leafs_array)
	if node.childs == []
		push!(leafs_array, node)
	else
		for i=1:length(node.childs)
			get_leafs(node.childs[i], leafs_array)
		end
	end
end

## penultimate_nodes
## Find the penultimate nodes of a tree
function get_penultimate_nodes(node::PathsTree, penult_nodes::Array{PathsTree,1})
	if (node.childs[1].childs==[])
		push!(penult_nodes,node)	
	else
		for(i=1:length(node.childs))
			get_penultimate_nodes(node.childs[i], penult_nodes)
		end
	end
end

## get_full_path_from_leaf
## Given a leaf, returns its full path backward recursivelly.
function get_full_path_from_leaf(leaf::PathsTree, path)
	push!(path, leaf.value)
	if leaf.level!=1
		get_full_path_from_leaf(leaf.father, path)		
	end
end

## get_all_paths
## Print all possible paths of a tree in a matrix form
function get_all_paths(node::PathsTree)
	leafs_array = PathsTree[]
	get_leafs(node, leafs_array)
	for i=1:length(leafs_array)
		path = Array[]
		get_full_path_from_leaf(leafs_array[i], path)
		println(join(reverse(path)',";"))
	end
end

## get_all_paths
## Load a matrix with all possible paths of a tree
function get_all_paths(node::PathsTree, M::Array)
	leafs_array = PathsTree[]
	get_leafs(node, leafs_array)
	for i=1:length(leafs_array)
		path = Any[]
		get_full_path_from_leaf(leafs_array[i], path)
		path_ = [collect(x) for x=reverse(path)]
		push!(M,path_)
		println(join(path_,";"))
	end
end

## get_full_probs_from_leaf
## Given a leaf, returns the probabilities of each node in the full path path backward recursivelly.
function get_full_probs_from_leaf(leaf::PathsTree, probs)
	push!(probs, leaf.prob)
	if leaf.level!=1
		get_full_probs_from_leaf(leaf.father, probs)
	end
end

## get_all_probs
## Print all possible nodes probabilities in a matrix form 
function get_all_probs(node::PathsTree)
	leafs_array = PathsTree[]
	get_leafs(node, leafs_array)
	for i=1:length(leafs_array)
		probs = Float64[]
		get_full_probs_from_leaf(leafs_array[i], probs)
		println(join(reverse(probs)',";"))
	end
end

###############################################################
### "Kernel" of Stochastic Approximation Method
###############################################################

## peteleco
## Function to apply the stochastic approximation method (aka, "peteleco" method) at one single stage
function peteleco(root::PathsTree, new_path, level, ak, dist)

	if root.childs != []
	
		#Find the nearest element between childs
		value = new_path[level,:]
		i, min_distance = get_nearest_child(root, value, dist)
		
		#Update the counters
		root.childs[i].counter = root.childs[i].counter + 1
		
		#Update the probabilities
		total = sum([x.counter for x = root.childs])
		for j=1:length(root.childs)
			root.childs[j].prob = root.childs[j].counter / total
		end
		
		#Peteleco... (each dimension a time)
		for q=1:length(value) 
			d = root.childs[i].value[q] - value[q]
			root.childs[i].value[q] = root.childs[i].value[q] - ak * abs(d) * sign(d)
		end

		#Run recursivelly on the nearest child node	
		peteleco(root.childs[i], new_path, level+1, ak, dist)

	end	

end

## lots_of_petelecos
## Function to apply the stochastic approximation method (aka, "peteleco" method) recursivelly
## starting from at root node. It works creating M new paths and using it to do the "petelecos"
## in the original tree.
function lots_of_petelecos(root::PathsTree, M, number_of_childs, path_function::Function, path_function_params, dist)
	S0 = root.value
	am = 3./(30+[1:M].^0.75) #Magic sequence!
	for i=1:M
		new_path = path_function(root, length(number_of_childs)-1, path_function_params)
		peteleco(root, new_path, 2, am[i], dist)
	end
end


