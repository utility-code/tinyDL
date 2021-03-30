# Train config
lr = 0.001
numEpochs = 1
afterEvery = 1
testSize = .1
verbose = True
pbarLength = 50
usegpu = False
# CE, MSE
lossfunc = "CE"
# GD, ADAM, SGD
optim = "GD" 

# dropout
activationdropout = False
actdropoutprob = .8
layerdropout = True
layerdropoutprob = .5 #between .5 to .8

# Plots
plotLoss = False
plotAcc = False

# Logs
log = False
logAfter = 10
logdir = "./experiments/"
