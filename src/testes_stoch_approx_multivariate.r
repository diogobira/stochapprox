#setwd("J:/GERIM3/ALM/ReducaodeCenarios/")
setwd("/home//diogo/git/stochapprox/src/")

O = read.csv(file="original_paths.csv", sep=";", header=FALSE)
N = read.csv(file="new_paths.csv", sep=";", header=FALSE)

## Probs
Nprobs =  read.csv(file="new_paths_probs.csv", sep=";", header=FALSE)
Nprobs = matrix(as.numeric(as.matrix(Nprobs)),ncol=length(Nprobs))
scen_prob = apply(Nprobs, 1, prod)
sum(scen_prob)

### 
coord = 2
O1 = O[O$V1==coord,-1]
O1 = t(matrix(as.numeric(as.matrix(O1)),ncol=length(O1)))
matplot(x=seq(1:nrow(O1)), y=O1, type="l")
N1 = N[N$V1==coord,-1]
N1 = t(matrix(as.numeric(as.matrix(N1)),ncol=length(N1)))
matplot(x=seq(1:nrow(N1)), y=N1, type="l")

exp(-0.15)*sum(apply(tail(N1,n=1)-11000,2 ,max,0) * scen_prob)
exp(-0.10)*sum(apply(tail(N1,n=1)-11,2 ,max,0) * scen_prob)

library(RQuantLib)
EuropeanOption(type="c", underlying=10000, strike=11000, 
               dividendYield=0, riskFreeRate=0.15, maturity=1, volatility=0.3)

EuropeanOption(type="c", underlying=10, strike=11, 
               dividendYield=0, riskFreeRate=0.10, maturity=1, volatility=0.25)
