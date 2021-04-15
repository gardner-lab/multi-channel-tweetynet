# Multi-Channel TweetyNet

## Overview

This repository contains a second implementation of **TweetyNet**, a convolutional neural network that processes spectrogram images of recorded bird song and assigns a syllable label to each discrete time step [[1]](#1). In other words, TweetyNet automates annotation of birdsong by syllable type. 

Our work extends the original project by (1) validating network performance in the face of significant noise and (2) supporting multi-channel input to the network where each channel is a spectrogram representing an identical audio sample but is constructed with varying time-frequency resolution. Since, in this implementation, the input channels can have different dimensions we modify TweetyNet's first layer to convolve and pool each channel to common size if necessary. Below is a diagramatic representation of TweetyNet sourced from the original publication. This diagram accurately represents the architecture of our implementation (less changes like input dimensions) when the network is processing typical (non multi-channel) input [[1]](#1).

<br>
<img src="https://github.com/gardner-lab/multi-channel-tweetynet/blob/master/figs/tweetynet.jpg?raw=true" href="#">
<br>


## Dataset

Tweetynet is a fully supervised the network that learns from labelled training data. In our experiments we exclusively relied on a publically available, labeled dataset of Bengalese Finch song [[2]](#2). To support usage of Multi-TweetyNet we also provide software to generate spectrograms and corresponding labels. The utilities for generating spectrograms are easily applied to novel audio datasets by making simple modifications, subject to the encoding of the source audio. Similarly, the software can be easily modified to support generating labels from novel annotation schemas. 

To explore network performance with noisy audio we added gaussian noise to recordings from the aforementioned dataset. Images below offer an example of a noiseless and noisy spectrogram of 1 second of recorded song.
<img src="https://github.com/gardner-lab/multi-channel-tweetynet/blob/master/figs/spectnonoise.png?raw=true" width=1000px height=400px>
<img src="https://github.com/gardner-lab/multi-channel-tweetynet/blob/master/figs/spectsnr0.5.png?raw=true" width=1000px height=400px>

<br>

## Results

Our results validate previous reports from [[1]](#1). In addition, we demonstrate TweetyNet's sensitivity to input representation (particularly in the presence of noise) and the success of a multi-channel, "wideband" input. The later result comes with caveats discussed below. In the figures shown below the legend labels indicate the FFT size used to construct inputs to the network. The labels "Wideband" and "Extreme-Wideband" correspond to networks processing multi-channel inputs, each composed of input representations derived from FFT sizes of 256, 512, 1024 and 64, 128, 256, 512, 1024, 2048, 4096, respectively.

<img src="https://github.com/gardner-lab/multi-channel-tweetynet/blob/master/figs/accnoise.png">

<img src="https://github.com/gardner-lab/multi-channel-tweetynet/blob/master/figs/accnonoise.png">

<br> 

## Usage

To get started, clone this repository. ```/env/tweetynet.yaml``` is a file specifying a conda environment containing all dependencies. There are five fundamental pieces of source code found in ```main/src/``` and corresponding parameter specifications found in ```main/parameters/```. Broadly speaking, the workflow to use this software from scratch is as follows:

1. Collect the dataset from [[2]](#2)
2. Install and activate the virtual environment. This can be done with ```conda env create -f tweetynet.yml``` You will need to install anaconda or miniconda if you have not yet. 
3. Specify global parameters in ```/main/parameters/params.py```
4. Parameterize ```/main/src/spect_writer.py``` via ```/main/parameters/spect_writer_params.py``` in order to generate spectrograms
5. Parameterize ```/main/src/labelvec_writer.py``` via ```/main/parameters/labelvec_writer_params.py``` in order to generate labels
6. Tune the network architecture via ```/main/parameters/network_params.py```
7. Train the network. 

For demo detailing the full work flow described above see the python notebook ```/main/demo.ipynb```.

<br>

## References
<a id="1">[1]</a> 
https://www.biorxiv.org/content/10.1101/2020.08.28.272088v1.full

<a id="2">[2]</a>
https://figshare.com/articles/dataset/Bengalese_Finch_song_repository/4805749

<br>

### Aside: Motivation and Details for Multi-Channel

Our re-implementation of TweetyNet was, in part, motivated by the hypothesis that we may be able to improve performance by varying the relative time-frequency resolution of input spectrograms. Applying the STFT to a discrete time signal we can generate a frequency-over-time representation of the signal. The critical parameters of this transformation include the FFT size and the window size, which dictate the number of discrete frequency bins to be computed and the number of samples over which to compute the sliding FFT, respectively. Since the FFT frequency resolution is determined by the length of the signal being analyzed and the sampling rate there exists a fundamental tradeoff between time and frequency resolution in the STFT.

To our knowledge there has been no systematic evaluation of the representational power of Bengalese Finch spectrograms computed with different time-frequency resolutions. Incidentally, Bengalese Finch song demonstrates complex temporal dynamics with variability in the length of both individual syllables and periods of non-singing. With this in mind, our goal was two-fold: evaluate how sensitive TweetyNet's performance is to variation in input representation time-frequency resolution and determine if the incorporation of less commmonly used representations into multi-channel inputs could result in improved performance. 

Below we show a detailed schematic of the first convolution + max pooling layer of the network processing a multi-channel input. In effect, we process each channel independently with a convolutional and max-pooling kernel with an appropriate aspect ratio. The convolution is input size preserving; however, the max-pooling operation downsamples each channel to identical dimensions so the channels can be stacked before being processed by the second convolution-pooling layer.  

<br>
<img src="https://github.com/gardner-lab/multi-channel-tweetynet/blob/master/figs/frontend.png?raw=true">
<br>

___F___ and ___T___ correspond to the number of frequency and time bins in the channel containing the spectrogram with the greatest frequency resolution, which we will refer to as ***S***. ___D___ is the number of filters used for each channel. ___CK___ and ___PK___ correspond to the convolution and max-pooling kernel size (subscripted by ___W___ and ___H___ for height and width). 

Simply put, we did not find a reliable boost in performance when processing multi-channel input; however, one of the multi-channel configurations we studied performed approximately as well as the best single-channel inputs. Directions for future research could involve rigorously evaluating the contribution from particular input channels in the multi-channel approach to determine whether a multi-channel approach carries utility for domains where the best FFT parameterization is not known apriori.
