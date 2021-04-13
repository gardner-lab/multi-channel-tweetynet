"""SpectWriter Class PARAMETERS"""

from parameters.params import (
    AUDIO_DIR_PATH,
    AUDIO_FILE_FMT,
    SAMP_FREQ,
    WINDOWED_SPECTS_DIR_PATH,
    UNCUT_SPECTS_DIR_PATH,
    SPECT_FILE_FMT,
    WINDOW_DUR,
)

# Generate spectrograms for the following fft sizes,
# window lengths, and hop lengths. Note, in order for
# tweetynet to downsample multiple spectrogrm channels
# to identical dimensions the window length must equal
# the fft size and the hop lengths must remain relative
# multiples of 2.
STFT_N_FFTS = [64, 128, 256, 512, 1024, 2048, 4096]
STFT_WIN_LENS = [64, 128, 256, 512, 1024, 2048, 4096]
STFT_HOP_LENS = [4, 8, 16, 32, 64, 128, 256]

# Drop signal below and above specified frequency cutoffs
FREQ_CUTOFFS = [500, 10000]

NORMALIZE_SPECTS = True
STFT_POWER = 2

# Signal to noise ratio. Set to -1 for no added noise
SNR = 1

# To increase the number of training samples we can duplicate
# windows and add unique noise to each copy. Only applies if
# snr is not set to -1
NUM_WINDOW_COPIES = 1

# We extract spectrogram windows for training with a sliding window.
# Here we set the duration of the sliding window overlap relative
# to the duration of the window itself.
WINDOW_STRIDE_DUR = 0.03 * WINDOW_DUR

# Here we refer to the number of audio samples in the window stride
NUM_SAMPLES_IN_STRIDE = int(SAMP_FREQ * WINDOW_STRIDE_DUR)

# Number of samples in a window must be divisible by stft hop
# length parameter in order to cleanly hit our spectrogram channel
# sizes as multiples of two.
NUM_SAMPLES_IN_WINDOW = int(SAMP_FREQ * WINDOW_DUR) + (
    max(STFT_HOP_LENS) - (int(SAMP_FREQ * WINDOW_DUR) % max(STFT_HOP_LENS))
)

# Put it all together
SPECT_WRITER_PARAMS = {
    "audio_dir_path": AUDIO_DIR_PATH,
    "audio_file_fmt": AUDIO_FILE_FMT,
    "samp_freq": SAMP_FREQ,
    "windowed_spects_dir_path": WINDOWED_SPECTS_DIR_PATH,
    "uncut_spects_dir_path": UNCUT_SPECTS_DIR_PATH,
    "spect_file_fmt": SPECT_FILE_FMT,
    "stft_n_ffts": STFT_N_FFTS,
    "stft_win_lens": STFT_WIN_LENS,
    "stft_hop_lens": STFT_HOP_LENS,
    "freq_cutoffs": FREQ_CUTOFFS,
    "normalize_spects": NORMALIZE_SPECTS,
    "stft_power": STFT_POWER,
    "snr": SNR,
    "num_window_copies": NUM_WINDOW_COPIES,
    "num_samples_in_window": NUM_SAMPLES_IN_WINDOW,
    "num_samples_in_stride": NUM_SAMPLES_IN_STRIDE,
}
