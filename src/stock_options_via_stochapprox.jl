###################################################################
# Stock Option Pricing Functions
# 
#
###################################################################

### price_european_call
### Price an European Call using a Paths Tree
###
function price_european_call(node, K, discountFactor, j)
	leafs = PathsTree[]
	get_leafs(node, leafs)

	## Calculate the payoff for each tree leaf	
	for i=1:length(leafs)
		S = leafs[i].value[j]
		leafs[i].optionValue = max(S-K,0)
		#println(S)
		#println(S,"---",leafs[i].optionValue)
	end

	## Find the penultimate nodes to start the backward update process
	penult_nodes = PathsTree[]
	penultimate_nodes(node, penult_nodes)
	backward_update(penult_nodes, discountFactor, "European")

	## Return the payoff
	node.optionValue
end


