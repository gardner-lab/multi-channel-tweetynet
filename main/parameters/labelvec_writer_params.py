"""LabelVecWriter Class Parameters"""

from parameters.params import (
    ANNOT_DIR_PATH,
    ANNOT_FILE_FMT,
    SPECT_FILE_FMT,
    LABELVEC_FILE_FMT,
    WINDOWED_SPECTS_DIR_PATH,
    WINDOWED_LABELVECS_DIR_PATH,
    UNCUT_SPECTS_DIR_PATH,
    UNCUT_LABELVECS_DIR_PATH,
    NO_LABEL_SYM,
    LABELSET,
    SAMP_FREQ,
)

from parameters.spect_writer_params import (
    NUM_SAMPLES_IN_WINDOW,
    NUM_SAMPLES_IN_STRIDE,
)

# Create labelmap (label --> int) from labelset
LABELMAP = dict((l, i) for i, l in enumerate(LABELSET))

LABELVEC_WRITER_PARAMS = {
    "annot_dir_path": ANNOT_DIR_PATH,
    "annot_file_fmt": ANNOT_FILE_FMT,
    "spect_file_fmt": SPECT_FILE_FMT,
    "labelvec_file_fmt": LABELVEC_FILE_FMT,
    "windowed_spects_dir_path": WINDOWED_SPECTS_DIR_PATH,
    "windowed_labelvec_dir_path": WINDOWED_LABELVECS_DIR_PATH,
    "uncut_spects_dir_path": UNCUT_SPECTS_DIR_PATH,
    "uncut_labelvec_dir_path": UNCUT_LABELVECS_DIR_PATH,
    "num_samples_in_window": NUM_SAMPLES_IN_WINDOW,
    "num_samples_in_stride": NUM_SAMPLES_IN_STRIDE,
    "samp_freq": SAMP_FREQ,
    "labelmap": LABELMAP,
    "no_label_sym": NO_LABEL_SYM,
}
