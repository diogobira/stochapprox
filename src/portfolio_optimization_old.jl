###################################################################
# portfolio_optimization.jl
# Multistage Portfolio Optimization using Stochastic Approximation
# and Simulated Annealing.
###################################################################

############################################
### Data Structures
############################################

### PathsTree Data Structure
type PathsTree

	## Attributes
	value::Float64
	level::Int32
	childs::Array{PathsTree,1}
	father::PathsTree
	counter::Int64
	prob::Float64
	portfolioValue::Float64
	
	## Constructors
	
	# Only for the root node 
	function PathsTree(value,level) 
		x = new() 
		x.value = value
		x.level = level
		x.childs = []
		x.counter = 0
		x.prob = 1
		x.portfolioValue = 1
		x
	end
	
	# For generic nodes
	function PathsTree(value, level, father::PathsTree) 
		x = new() 
		x.value = value
		x.level = level
		x.childs = []
		x.father = father
		x.counter = 0
		x.prob = 0
		x.portfolioValue = 1
		x
	end
	
end

############################################
### Auxiliary Functions
############################################

## call_path
## Function to build a path (or just the next value, when N=1) of intrinsic values of an call option
function stock_path(S0, N, params)

	#Separate the params
	r = params[:r] #risk-free rate
	sigma = params[:sigma] #vol
	dt = params[:dt] #Step size
	K = params[:K] #strike
	
	#Array for the returning path
	path = zeros(N+1)
	path[1]=S0
	for i=2:N+1
		path[i] = path[i-1]*exp((r-0.5*sigma^2)*dt +  sqrt(dt) * randn(1)[1])
	end
	
	#If N=1, return only
	if(N>1)
		path
	elseif(N==1)
		path[2]
	end

end

###############################################################
### Auxiliary Tree Creation/Manipulation/Search Functions
###############################################################

### add_childs
### Function to build a tree a given specific function/params and the given number of nodes at each level
function add_childs(node, payofffun::Function, number_of_childs, path_function::Function, path_function_params)
	for i=1:number_of_childs[node.level]
		value = path_function(node.value, 1, path_function_params)
		child_node = PathsTree(value, payofffun, node.level+1, node)
		push!(node.childs, child_node)
		add_childs(child_node, payofffun, number_of_childs, path_function, path_function_params)
	end	
end

## get_nearest_child
## Function to get the index of child with (euclidean) minimum distance to the father
function get_nearest_child(node, value)
	distances = [norm(n.value - value) for n = node.childs]
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
		path = Float64[]
		get_full_path_from_leaf(leafs_array[i], path)
		println(join(reverse(path)',";"))
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
function peteleco(root::PathsTree, new_path, level, ak)
	if root.childs != []
	
		#Find the nearest element between childs
		value = new_path[level]
		i, min_distance = get_nearest_child(root, value)
		
		#Update the counters
		root.childs[i].counter = root.childs[i].counter + 1
		
		#Update the probabilities
		total = sum([x.counter for x = root.childs])
		for j=1:length(root.childs)
			root.childs[j].prob = root.childs[j].counter / total
		end
		
		#Peteleco...
		if root.childs[i].value > value
			root.childs[i].value = root.childs[i].value - ak * min_distance * (1)
		else
			root.childs[i].value = root.childs[i].value - ak * min_distance * (-1)
		end
		peteleco(root.childs[i], new_path, level+1, ak)
		
	end	
end

## lots_of_petelecos
## Function to apply the stochastic approximation method (aka, "peteleco" method) recursivelly
## starting from at root node. It works creating M new paths and using it to do the "petelecos"
## in the original tree.
function lots_of_petelecos(root::PathsTree, M, number_of_childs, path_function::Function, path_function_params)
	S0 = root.value
	am = 3./(30+[1:M].^0.75) #Magic sequence!
	for i=1:M
		new_path = path_function(S0, length(number_of_childs)-1, path_function_params)
		peteleco(root::PathsTree, new_path, 2, am[i])
	end
end

###############################################################
### Price Functions
###############################################################

## penultimate_nodes
## Find the penultimate nodes of a tree, from which the backward induction
## will start when we are pricing
function penultimate_nodes(node::PathsTree, penult_nodes::Array{PathsTree,1})
	if (node.childs[1].childs==[])
		push!(penult_nodes,node)	
	else
		for(i=1:length(node.childs))
			penultimate_nodes(node.childs[i], penult_nodes)
		end
	end
end

## update_value
## Given a node, update its value if the discounted expectation is greater than
## the node current node value.
function update_value(node::PathsTree, discounFactor)
	disc_expectation = discounFactor * sum([x.portfolioValue*x.prob for x=node.childs])
	if disc_expectation >= node.portfolioValue
		node.portfolioValue = disc_expectation
		node.exerciseFlag = false
	else
		node.exerciseFlag = true
	end 
end

## backward_update
## Starting from a list with the penultimate nodes, update
## the node values and run recursivelly for the parent nodes
function backward_update(nodes, discounFactor)

	# If we already reach the root node
	# update the value and return
	if length(nodes)==1
		update_value(nodes[1], discounFactor)
		return
	end	

	# If we are not at the root node,
	# update the value an call the function for the list nodes
	# at the immediate early stage of the tree
	parent_nodes = PathsTree[]		
	for i=1:length(nodes)
		update_value(nodes[i], discounFactor)
		if i==1
			push!(parent_nodes, nodes[i].father)
		elseif last(parent_nodes)!=nodes[i].father
			push!(parent_nodes, nodes[i].father)
		end
	end
	backward_update(parent_nodes, discountFactor)

end

###############################################################
### Some tests...
###############################################################

## Creating of a call option intrisic values path
params = {:r=>0.1, :sigma=>0.3, :dt=>10/252, :K=>110}
stock_path(100, 10, params)
calloption_110(s) = max(s-110,0) 

## Creating a tree of paths of a call option intrisic values with 10 time steps
## with two ramification at node 
number_of_childs = push!([2 for i=1:10],0)
root = PathsTree(100,calloption_110, 1)
add_childs(root, calloption_110, number_of_childs, stock_path, params) 	

## Creating a copy of original tree and running the stochastic approximation method
new_root = deepcopy(root);
lots_of_petelecos(new_root, 100000, number_of_childs, stock_path, params);

##
penult_nodes = PathsTree[]
penultimate_nodes(new_root, penult_nodes)
discountFactor = exp(-0.1*10/252)
backward_update(penult_nodes, discountFactor)

#get_all_paths(root)
#get_all_paths(new_root)
#get_all_probs(new_root)
