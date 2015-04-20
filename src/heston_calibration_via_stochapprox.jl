###################################################################
# Implied Heston Model Calibration via Multinomial Tree  
# and Stochastic Approximation
###################################################################

function heston_path (node, N, params)

	#Separate the params
	mu = params[:mu]
	v0 = params[:v0] 
	kappa = params[:kappa]
	theta = params[:theta]
	sigma = params[:sigma]
	rho = params[:rho]
	dt = params[:dt] 

	#Get the dimension off the process
	d = size(node.value)[2]

	#Create the correlated "shocks"
	A = chol(rho)
	shocks = (A' * randn(d,N))'

	#Volatilities
	#vol = sqrt(diag(sigma))'

	#Array for the paths
	path = zeros(N+1,d)
	path[1,:] = node.value
	sdt = sqrt(dt)
	for i=2:N+1
		### Variance Path	
		path[i,1] = path[i-1,1] + kappa * (theta - path[i-1,1]) * dt + sigma * sqrt(path[i-1,1]) * sdt * shocks[i-1,1] 
		path[i,1] = max(path[i,1],0)
		### Price Path
		path[i,2] = path[i-1,2] + path[i-1,2] * mu * dt + path[i-1,2] * sqrt(path[i,1]) * sdt * shocks[i-1,2] 
	end
	
	#If N==1, return only the next step values. Otherwise, returns the complete 
	#path including the initial values
	N==1 ? path[2,:] : path

end


