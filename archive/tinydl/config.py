
# Train config
lr = 0.001
<<<<<<< HEAD:archive/tinydl/config.py
numEpochs = 1
=======
numEpochs = 50
>>>>>>> 508dd746680bc19abe6951b886c53fc83bb16d2d:tinydl/config.py
afterEvery = 1
testSize = .1
verbose = True
pbarLength = 50
usegpu = False
<<<<<<< HEAD:archive/tinydl/config.py
# CE, MSE
lossfunc = "CE"
# GD, ADAM, SGD
optim = "GD" 
=======
# None or number
batchsize = 64
# CE, MSE, SVM
lossfunc = "SVM"
# ADAM, SGD, GD
optim = "SGD" 
>>>>>>> 508dd746680bc19abe6951b886c53fc83bb16d2d:tinydl/config.py

# dropout
activationdropout = False
actdropoutprob = .8
layerdropout = False
layerdropoutprob = .5 #between .5 to .8

# Logs
log = False
logAfter = 10
logdir = "./experiments/"
