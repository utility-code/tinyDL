# TRAINING DETAILS
numEpochs = 100
batch_size = 64
learning_rate = 0.01
# LOG
log = True
logdir = "./experiments/"
afterEvery = 10

# PLOTS
plotLoss = False

#LOSS
# MSE, 
lossFunction = "MSE"

# OPTIMIZER

optimizer = "ADAM"
# GD
#  optimizer = "GD"

# GDM, NGD
momentum = .9 

# RMS Prop
#  decay = .9
#  eps = 1e-10

# ADAM
beta1 = .9
beta2 = .999
eps = 1e-8
