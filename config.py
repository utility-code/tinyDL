# Train config
lr = 0.001
numEpochs = 1000
afterEvery = 100
testSize = .1
verbose = True
pbarLength = 50
# CE, MSE
lossfunc = "MSE"
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
