#setwd("J:/GERIM3/ALM/ReducaodeCenarios/")
setwd("/home//diogo/git/stochapprox/src/")

install.packages("NMOF")
library(NMOF)

O = read.csv(file="original.heston.path.csv", sep=";", header=FALSE)
N = read.csv(file="heston.path.csv", sep=";", header=FALSE)

O = read.csv(file="original.gbm.path.csv", sep=";", header=FALSE)
N = read.csv(file="gbm.path.csv", sep=";", header=FALSE)

## Probs
#Nprobs =  read.csv(file="new_paths_probs.csv", sep=";", header=FALSE)
#Nprobs = matrix(as.numeric(as.matrix(Nprobs)),ncol=length(Nprobs))
#scen_prob = apply(Nprobs, 1, prod)
#sum(scen_prob)

### 
coord = 2
O1 = O[O$V1==coord,-1]
O1 = t(matrix(as.numeric(as.matrix(O1)),ncol=length(O1)))
matplot(x=seq(1:nrow(O1)), y=O1, type="l")
N1 = N[N$V1==coord,-1]
N1 = t(matrix(as.numeric(as.matrix(N1)),ncol=length(N1)))
matplot(x=seq(1:nrow(N1)), y=N1, type="l")

callHestoncf(
  S=100, X=110, tau=1, r=0.1, q=0, v0=0.35^2, vT=0.45^2, 
  rho=-0.4, sigma=0.3, k=63/252)

EuropeanCall(S0=100, X=110, r=0.1, tau=1, sigma=0.3)
