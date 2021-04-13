from parameters.params import (
    WINDOWED_SPECTS_DIR_PATH,
    WINDOWED_LABELVECS_DIR_PATH,
    UNCUT_SPECTS_DIR_PATH,
    UNCUT_LABELVECS_DIR_PATH,
    SPECT_FILE_FMT,
    LABELVEC_FILE_FMT,
    AUDIO_FILE_FMT,
)

from parameters.net_params import N_FFTS

TRAIN_DATASET_PARAMS = {
    "spect_files_dir_path": WINDOWED_SPECTS_DIR_PATH,
    "labelvec_files_dir_path": WINDOWED_LABELVECS_DIR_PATH,
    "spect_file_fmt": SPECT_FILE_FMT,
    "labelvec_file_fmt": LABELVEC_FILE_FMT,
    "audio_file_fmt": AUDIO_FILE_FMT,
    "network_nffts": N_FFTS,
}

EVAL_DATASET_PARAMS = {
    "spect_files_dir_path": UNCUT_SPECTS_DIR_PATH,
    "labelvec_files_dir_path": UNCUT_LABELVECS_DIR_PATH,
    "spect_file_fmt": SPECT_FILE_FMT,
    "labelvec_file_fmt": LABELVEC_FILE_FMT,
    "network_nffts": N_FFTS,
}
