# TRAINING DETAILS
numEpochs = 100
batch_size = 1000
learning_rate = 0.01
# LOG
log = True
logdir = "./experiments/"
afterEvery = 10

# PLOTS
plotLoss = False

# LOSS
# MSE,
lossFunction = "MSE"

# OPTIMIZER

optimizer = "ADAM"
# GD
#  optimizer = "GD"

# GDM, NGD
momentum = 0.9

# RMS Prop
#  decay = .9
#  eps = 1e-10

# ADAM
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
