"""Training Parameters"""

from parameters.params import DEVICE

NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 1
NUM_WORKERS = 8
EVAL_STEP = TRAIN_BATCH_SIZE * 13
LR = 0.001
NUM_RUNS = 40
