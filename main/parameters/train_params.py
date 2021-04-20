"""Training Parameters"""

from parameters.params import DEVICE


# Network converges quickly
NUM_EPOCHS = 3


# Small batch size tends to work well
TRAIN_BATCH_SIZE = 8


# EvalDataset elements are full songs batched into training
# window pieces. So a single element corresponds to a larger batch.
EVAL_BATCH_SIZE = 1


# Modify subject to your hardware
NUM_WORKERS = 8


# Perform evaluation after this many training windows
EVAL_STEP = TRAIN_BATCH_SIZE * 13


# Learning rate
LR = 0.001


# Number of new instantiations of network to run
NUM_RUNS = 10
