"""MultiChannelTweetynet Class Parameters"""

from parameters.params import DEVICE, NUM_CLASSES

# All elements marked with '****' will be filled programatically below
# It is not advised to edit those fields manually
NETWORK_PARAMS = {
    "device": DEVICE,
    "num_classes": NUM_CLASSES,
    "n_ffts": [],  # ****
    "conv1_pad_same": True,
    "num_conv1_filters": {},  # ****
    "conv1_kernel_sizes": {},  # ****
    "conv1_stride_lens": {},  # ****
    "pool1_pad_same": False,
    "pool1_kernel_sizes": {},  # ****
    "pool1_stride_lens": {},  # ****
    "num_conv2_filters": 64,
    "conv2_kernel_size": (3, 3),
    "conv2_stride_len": (1, 1),
    "conv2_pad_same": True,
    "pool2_kernel_size": (2, 1),
    "pool2_stride_len": (2, 1),
    "pool2_pad_same": False,
}

# Specify the spectral "channels" the network will process in sorted order
# Ensure that you have in fact generated spectrograms for all the FFT sizes provided below
N_FFTS = [256, 512, 1024]

# This must be a multiple of the number of input channels (i.e. len(N_FFTS))
NUM_CONV1_FILTERS = 42

# NOTE: for any selection of 'n_ffts' it makes sense that the kernel sizes are
# 2x multiples of one another relative to the factor relationship between the fft
# parameters themselves. The same rule applies for pool1 parameters specified below.
# NOTE: having observed poor performance we comment out parameter sets for FFT sizes
# 64, 128, 2048, and 4096. Including them at all is a reminder that this software can 
# accomodate different FFT sizes provided a proper parameterization. 
CONV1_KERNEL_SIZES = {
    #64: (2, 64),
    #128: (4, 32),
    256: (4, 4),
    512: (8, 2),
    1024: (16, 1),
    #2048: (64, 2),
    #4096: (128, 1),
}

# NOTE: typically we do not extend the first convolution's stride to reserve
# downsampling for the subsequent pooling operations
CONV1_STRIDE_LENS = {
    #64: (1, 1),
    #128: (1, 1),
    256: (1, 1),
    512: (1, 1),
    1024: (1, 1),
    #2048: (1, 1),
    #4096: (1, 1),
}

POOL1_KERNEL_SIZES = {
    #64: (2, 64),
    #128: (4, 32),
    256: (4, 4),
    512: (8, 2),
    1024: (16, 1),
    #2048: (64, 2),
    #4096: (128, 1),
}

POOL1_STRIDE_LENS = {
    #64: (2, 64),
    #128: (4, 32),
    256: (4, 4),
    512: (8, 2),
    1024: (16, 1),
    #2048: (64, 2),
    #4096: (128, 1),
}

# Programatically fill missing entries in NETWORK_PARAMS subject to
# desired input channels
for nfft in N_FFTS:
    NETWORK_PARAMS["conv1_kernel_sizes"][nfft] = CONV1_KERNEL_SIZES[nfft]
    NETWORK_PARAMS["conv1_stride_lens"][nfft] = CONV1_STRIDE_LENS[nfft]
    NETWORK_PARAMS["pool1_kernel_sizes"][nfft] = POOL1_KERNEL_SIZES[nfft]
    NETWORK_PARAMS["pool1_stride_lens"][nfft] = POOL1_STRIDE_LENS[nfft]
    NETWORK_PARAMS["num_conv1_filters"][nfft] = NUM_CONV1_FILTERS // len(N_FFTS)
NETWORK_PARAMS["n_ffts"] = N_FFTS
