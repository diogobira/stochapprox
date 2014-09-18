#PathsTree Data Structure
type PathsTree
	value::Float64
	level::Int32
	childs::Array{PathsTree,1}
	father::PathsTree
	counter::Int64
	prob::Float64
	
	#Only for the root of the tree
	function PathsTree(value,level) 
		x = new() 
		x.value = value
		x.level = level
		x.childs = []
		x.counter = 0
		x.prob = 1
		x
	end
	
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

#Function to build a PathsTree using a GBM and a specific number of nodes at each level
function add_childs(node, r, sigma, dt, number_of_childs)
	for i=1:number_of_childs[node.level]
		value = node.value * exp((r-0.5*sigma^2)*dt +  sqrt(dt) * randn(1)[1])
		child_node = PathsTree(value, node.level+1, node)
		#child_node = PathsTree(value, node.level+1)
		push!(node.childs, child_node)
		add_childs(child_node, r, sigma, dt, number_of_childs)
	end	
end

#Function to print a PathsTree
import Base.print
function print(p::PathsTree)
	println(repeat(";",p.level-1),p.value)
	for i=1:length(p.childs)
		print(p.childs[i])
	end
end

#Function to create a new path 
function create_path(S0, r, sigma, dt, N)
	path = zeros(N+1)
	path[1]=S0
	for i=2:N+1
		path[i] = path[i-1]*exp((r-0.5*sigma^2)*dt +  sqrt(dt) * randn(1)[1])
	end
	path
end

#Function to get the index of child with min dist(child.value - new_path.value)
function get_nearest_child(node, value)
	distances = [sqrt((n.value - value)^2) for n = node.childs]
	min_distance = minimum(distances)
	idx = findfirst(distances, minimum(distances))
	return idx, min_distance
end

#Function to apply the "peteleco" at each
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

#Main petelecos function
function lots_of_petelecos(root::PathsTree, K, number_of_childs)
	S0, r, sigma, dt, N = 100, 0.1, 0.5, 63/252, 4
	ak = 3./(30+[1:K].^0.75)
	for i=1:K
		new_path = create_path(S0, r, sigma, dt, length(number_of_childs)-1)
		peteleco(root::PathsTree, new_path, 2, ak[i])
	end
end

#Get all the leafs of a PathsTree
function get_leafs(node::PathsTree, leafs_array)
	if node.childs == []
		push!(leafs_array, node)
	else
		for i=1:length(node.childs)
			get_leafs(node.childs[i], leafs_array)
		end
	end
end

#Given a leaf, print its full path
function get_full_path_from_leaf(leaf::PathsTree, path)
	push!(path, leaf.value)
	if leaf.level!=1
		get_full_path_from_leaf(leaf.father, path)		
	end
end

#
function get_all_paths(node::PathsTree)
	leafs_array = PathsTree[]
	get_leafs(node, leafs_array)
	for i=1:length(leafs_array)
		path = Float64[]
		get_full_path_from_leaf(leafs_array[i], path)
		println(join(reverse(path)',";"))
	end
end

#
function get_full_probs_from_leaf(leaf::PathsTree, probs)
	push!(probs, leaf.prob)
	if leaf.level!=1
		get_full_probs_from_leaf(leaf.father, probs)
	end
end

# 
function get_all_probs(node::PathsTree)
	leafs_array = PathsTree[]
	get_leafs(node, leafs_array)
	for i=1:length(leafs_array)
		probs = Float64[]
		get_full_probs_from_leaf(leafs_array[i], probs)
		println(join(reverse(probs)',";"))
	end
end


#Building an example of a PathsTree using a GBM
#require("D:/Documents and Settings/digob/Desktop/tmp/paths_tree_v2.jl")
#number_of_childs = [2,2,2,2,0];
#number_of_childs = [2,3,4,5,0];
#root = PathsTree(100,1);
#add_childs(root, 0.1, 0.5, 63/252, number_of_childs);	
#new_root = deepcopy(root);
#lots_of_petelecos(new_root, 1000, number_of_childs)
#get_all_paths(root)
#get_all_paths(new_root)
#get_all_probs(new_root)