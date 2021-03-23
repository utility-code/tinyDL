# Train config
lr = 0.001
numEpochs = 50
afterEvery = 1
testSize = .1
verbose = True
pbarLength = 50
usegpu = False
# None or number
batchsize = 64
# CE, MSE, SVM
lossfunc = "SVM"
# ADAM, SGD
optim = "SGD" 

# dropout
activationdropout = False
actdropoutprob = .8
layerdropout = False
layerdropoutprob = .5 #between .5 to .8

# Plots
plotLoss = False
plotAcc = False

# Logs
log = False
logAfter = 10
logdir = "./experiments/"
