import torch
import pickle
from parameters.params import WINDOWED_SPECTS_DIR_PATH, SPECT_FILE_FMT
from parameters.net_params import NETWORK_PARAMS
from src.utilities import sample_train_eval_files
from src.dataset import EvalDataset, TrainDataset
from parameters.params import (
    WINDOWED_SPECTS_DIR_PATH,
    UNCUT_SPECTS_DIR_PATH,
    AUDIO_DIR_PATH,
    AUDIO_FILE_FMT,
    SPECT_FILE_FMT,
)
from parameters.dataset_params import EVAL_DATASET_PARAMS, TRAIN_DATASET_PARAMS
from parameters.train_params import (
    DEVICE,
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    NUM_WORKERS,
    EVAL_STEP,
    LR,
    NUM_RUNS,
)
from torch.utils.data import DataLoader
from src.model import TweetynetModel
from src.utilities import get_spect_window_shapes, get_dataset_dur
from src.network import MultiChannelTweetynet

N_FFTS = NETWORK_PARAMS["n_ffts"]

input_shapes = get_spect_window_shapes(WINDOWED_SPECTS_DIR_PATH, SPECT_FILE_FMT, N_FFTS)
print(input_shapes)
tweetynet = MultiChannelTweetynet(**NETWORK_PARAMS, input_shapes=input_shapes)
print(tweetynet)
del tweetynet

RESULT_PATH = "/home/luke/work/tweetynet/main/results/wrapup/snr0.5/4096_1000ms.pkl"
print(RESULT_PATH)

all_xs = []
all_accs = []
all_durs = []

for i in range(NUM_RUNS):

    print(f"RUN NUMBER: {i+1}")

    train_files_list, eval_files_list = sample_train_eval_files(
        UNCUT_SPECTS_DIR_PATH, 40, 10, AUDIO_FILE_FMT, SPECT_FILE_FMT,
    )
    train_set_dur = get_dataset_dur(train_files_list, AUDIO_DIR_PATH)
    eval_set_dur = get_dataset_dur(eval_files_list, AUDIO_DIR_PATH)
    all_durs.append((train_set_dur, eval_set_dur))

    train_dataset = TrainDataset(
        **TRAIN_DATASET_PARAMS, audio_files_list=train_files_list
    )
    eval_dataset = EvalDataset(**EVAL_DATASET_PARAMS, spect_files_list=eval_files_list)

    train_data = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )
    eval_data = DataLoader(
        eval_dataset,
        batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )

    tweetynet = MultiChannelTweetynet(**NETWORK_PARAMS, input_shapes=input_shapes)

    run_params = {
        "net": tweetynet,
        "train_data": train_data,
        "eval_data": eval_data,
        "num_epochs": NUM_EPOCHS,
        "eval_step": EVAL_STEP,
        "lr": LR,
    }

    model = TweetynetModel(device=DEVICE)
    xs, accs = model.run(**run_params)
    all_xs.append(xs)
    all_accs.append(accs)

with open(RESULT_PATH, "wb",) as handle:
    pickle.dump((all_xs, all_accs, all_durs), handle)
