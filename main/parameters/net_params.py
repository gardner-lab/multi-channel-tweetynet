"""MultiChannelTweetynet Class Parameters"""

from parameters.params import DEVICE, NUM_CLASSES

NETWORK_PARAMS = {
    "device": DEVICE,
    "num_classes": NUM_CLASSES,
    "n_ffts": [],
    "conv1_pad_same": True,
    "num_conv1_filters": {},
    "conv1_kernel_sizes": {},
    "conv1_stride_lens": {},
    "pool1_pad_same": False,
    "pool1_kernel_sizes": {},
    "pool1_stride_lens": {},
    "num_conv2_filters": 64,
    "conv2_kernel_size": (3, 3),
    "conv2_stride_len": (1, 1),
    "conv2_pad_same": True,
    "pool2_kernel_size": (2, 1),
    "pool2_stride_len": (2, 1),
    "pool2_pad_same": False,
}

# Specify the spectral "channels" the network will process in sorted order
N_FFTS = [64, 128, 256, 512, 1024, 2048, 4096]

# This must be a multiple of the number of input channels (i.e. len(n_nffts))
NUM_CONV1_FILTERS = 42

# NOTE: for any selection of 'n_ffts' it makes sense that the kernel sizes are
# 2x multiples of one another relative to the factor relationship between the fft
# parameters themselves. The same rule applies for pool1 parameters specified below.
CONV1_KERNEL_SIZES = {
    64: (2, 64),
    128: (4, 32),
    256: (8, 16),
    512: (16, 8),
    1024: (32, 4),
    2048: (64, 2),
    4096: (128, 1),
}

# NOTE: typically we do not extend the first convolution's stride to reserve
# downsampling for the subsequent pooling operations
CONV1_STRIDE_LENS = {
    64: (1, 1),
    128: (1, 1),
    256: (1, 1),
    512: (1, 1),
    1024: (1, 1),
    2048: (1, 1),
    4096: (1, 1),
}

POOL1_KERNEL_SIZES = {
    64: (2, 64),
    128: (4, 32),
    256: (8, 16),
    512: (16, 8),
    1024: (32, 4),
    2048: (64, 2),
    4096: (128, 1),
}

POOL1_STRIDE_LENS = {
    64: (2, 64),
    128: (4, 32),
    256: (8, 16),
    512: (16, 8),
    1024: (32, 4),
    2048: (64, 2),
    4096: (128, 1),
}


for nfft in N_FFTS:
    NETWORK_PARAMS["conv1_kernel_sizes"][nfft] = CONV1_KERNEL_SIZES[nfft]
    NETWORK_PARAMS["conv1_stride_lens"][nfft] = CONV1_STRIDE_LENS[nfft]
    NETWORK_PARAMS["pool1_kernel_sizes"][nfft] = POOL1_KERNEL_SIZES[nfft]
    NETWORK_PARAMS["pool1_stride_lens"][nfft] = POOL1_STRIDE_LENS[nfft]
    NETWORK_PARAMS["num_conv1_filters"][nfft] = NUM_CONV1_FILTERS // len(N_FFTS)
NETWORK_PARAMS["n_ffts"] = N_FFTS
