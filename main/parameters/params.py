"""Global Parameters"""
from src.utilities import get_labelset

# SPECIFY PATH TO AUDIO FILES
AUDIO_DIR_PATH = "/Users/luke/Desktop/032212/"

# SPECIFY PATH TO ANNOTATION FILES
ANNOT_DIR_PATH = "/Users/luke/Desktop/032212/"

# SPECIFY TARGET DIRECTORY FOR GENERATED SPECTROGRAMS
SPECT_DIR_PATH = "/Users/luke/Desktop/outdata/"

# SPECIFY SUBDIRECTORIES FOR SPECTROGRAM WINDOWS AND WINDOW LABELVECTORS
WINDOWED_SPECTS_DIR_PATH = SPECT_DIR_PATH + "windowed/"
WINDOWED_LABELVECS_DIR_PATH = WINDOWED_SPECTS_DIR_PATH + "labs/"

# SPECIFY SUBDIRECTORIES FOR UNCUT SPECTROGRAMS AND LABELVECTORS
UNCUT_SPECTS_DIR_PATH = SPECT_DIR_PATH + "uncut/"
UNCUT_LABELVECS_DIR_PATH = UNCUT_SPECTS_DIR_PATH + "labs/"

# Note this software exclusively supports spectrogram
# and associated labelvector generation for the following
# file types. See the following links for more information
#   https://figshare.com/articles/Bengalese_Finch_song_repository/4805749
#   https://github.com/NickleDave/evfuncs
AUDIO_FILE_FMT = ".cbin"
ANNOT_FILE_FMT = ".cbin.not.mat"

# Provide desired file suffix for labelvectors
LABELVEC_FILE_FMT = ".labvec"

# Provide desired file suffix for spectrograms
SPECT_FILE_FMT = ".spect"

# Audio recording specific sampling frequency
SAMP_FREQ = 32000

DEVICE = "cpu"

# Duration of windows processed by the network (in seconds)
WINDOW_DUR = 1

# Label for silence
NO_LABEL_SYM = "nolab"

# Collect all labels used in specified annotion directory. This method DOES NOT generalize beyond
# annotations provided in https://figshare.com/articles/Bengalese_Finch_song_repository/4805749.
LABELSET = get_labelset(ANNOT_DIR_PATH, ANNOT_FILE_FMT) + [NO_LABEL_SYM]

# Total number of possible classifications
NUM_CLASSES = len(LABELSET)
