import os
from glob import glob
import numpy as np
from evfuncs import load_cbin
import torch
import pickle
from datetime import datetime as dt
from scipy.signal import butter, lfilter
from torch import hann_window, stft
from torch.nn.functional import pad as pad_waveform
from numpy.random import normal


class SpectWriter:
    def __init__(
        self,
        audio_dir_path,
        audio_file_fmt,
        samp_freq,
        windowed_spects_dir_path,
        uncut_spects_dir_path,
        spect_file_fmt,
        stft_n_ffts,
        stft_hop_lens,
        stft_win_lens,
        freq_cutoffs,
        normalize_spects,
        stft_power,
        snr,
        num_window_copies,
        num_samples_in_window,
        num_samples_in_stride,
    ):
        self.audio_dir_path = audio_dir_path
        self.audio_file_fmt = audio_file_fmt
        self.samp_freq = samp_freq
        self.windowed_spects_dir_path = windowed_spects_dir_path
        self.uncut_spects_dir_path = uncut_spects_dir_path
        self.spect_file_fmt = spect_file_fmt
        self.max_hop_len = max(stft_hop_lens)
        self.stft_params = list(zip(stft_n_ffts, stft_win_lens, stft_hop_lens))
        self.freq_cutoffs = freq_cutoffs
        self.normalize_spects = normalize_spects
        self.stft_power = stft_power
        self.snr = snr
        if self.snr == -1:
            self.num_window_copies = 1
        else:
            self.num_window_copies = num_window_copies
        self.num_samples_in_window = num_samples_in_window
        self.num_samples_in_stride = num_samples_in_stride

    # SpectWriter entry point, main method.
    def write(self, audio_files_list=None):
        if audio_files_list is None:

            # Collect all audio files from provided audio directory
            audio_file_paths = glob(
                os.path.join(self.audio_dir_path, "*" + self.audio_file_fmt)
            )
        else:
            audio_file_paths = [self.audio_dir_path + f for f in audio_files_list]

        # For each audio file generate uncut (complete) and windowed spectrograms
        for i, afp in enumerate(audio_file_paths):
            audio_waveform = self.load_waveform(afp)

            # Pad audio s.t. the number of samples is divisible by window size.
            # Now can reshape spectrograms to batched window representation when needed.
            audio_waveform = self.pad_audio(audio_waveform, self.num_samples_in_window)

            # Compute "new" audio duration in milliseconds following pad
            audio_duration = (len(audio_waveform) / self.samp_freq) * 1000

            # Extract list of audio "windows" and associated start, end times
            audio_windows, times = self.extract_audio_windows_and_times(audio_waveform)

            # Add noise to audio if specified
            if self.snr != -1:
                audio_windows = self.add_noise(audio_windows)
                audio_waveform = self.add_noise([audio_waveform])

            # Generate windowed spectrograms. Also returns window lengths for ref
            # when computing uncut spectrogram and batching as windows.
            spect_windows, window_lengths = self.gen_spect_windows(audio_windows)

            # Generate uncut spects. Returned in two forms as 2d tensor and 3d tensor
            # of batched windows
            spects, batched_windows = self.gen_sepcts(audio_waveform, window_lengths)

            # Write uncut and windowed spects to disk
            filename = afp.split("/")[-1].split(self.audio_file_fmt)[0]
            self.windows_to_disk(filename, spect_windows, times)
            self.spects_to_disk(filename, spects, batched_windows, audio_duration)

    # Extract audio windows and start, stop times of each window
    def extract_audio_windows_and_times(self, audio_waveform):

        # pad audio s.t. the number of samples is divisible by the sliding
        # window stride duration
        audio_waveform = self.pad_audio(
            audio_waveform, modulus=self.num_samples_in_stride
        )

        total_num_samples = len(audio_waveform)

        # Starting index for final window (i.e. sliding window cutoff point)
        last_window_idx = total_num_samples - self.num_samples_in_window

        # Duration of window in milliseconds
        window_dur = (self.num_samples_in_window / self.samp_freq) * 1000

        windows = []
        times = []

        for i in range(0, last_window_idx, self.num_samples_in_stride):

            # Extract and save current window
            window = audio_waveform[i : i + self.num_samples_in_window]
            windows.append(window)

            # Compute and save start and end time of current window
            start_time = (i / self.samp_freq) * 1000
            time = (start_time, start_time + window_dur)
            times.append(time)

            # If we have duplicate windows from added noise duplicate saving time tuple
            if self.snr != -1:
                for j in range(self.num_window_copies - 1):
                    windows.append(window)
                    times.append(time)

        return windows, times

    # Generate uncut spectrograms and batched window representation over provided audio.
    # 'window_lengths': dictionary of spectrogram window lengths indexed by fft size
    def gen_sepcts(self, audio_waveform, window_lengths):

        # Number of unique, non-overlaping windows in audio
        num_windows = int(len(audio_waveform) / self.num_samples_in_window)

        spects = {}

        # For each stft parameter formulation compute a spectrogram and save it
        for n_fft, win_len, hop_len in self.stft_params:
            pad = n_fft // 2
            spect = self.spectrogram(audio_waveform, n_fft, win_len, hop_len, pad)
            spects[n_fft] = spect

        batched_windows = {}

        # Reshape and save computed spectrograms into 3d batched window representation
        for n_fft, spect in spects.items():
            window_length = window_lengths[n_fft]
            tmp = torch.empty((num_windows, spect.size()[0], window_length))
            for i in range(num_windows):
                tmp[i, :, :] = spect[:, i * window_length : (i + 1) * window_length]
            batched_windows[n_fft] = tmp

        return spects, batched_windows

    # Generate spectrogrm for each audio "window" in provided list.
    def gen_spect_windows(self, audio_windows):
        spect_windows = []

        # For each audio window
        for aw in audio_windows:
            spects = {}
            # Compute and save spectrogram for each stft param formulation
            for n_fft, win_len, hop_len in self.stft_params:
                pad = n_fft // 2
                spect = self.spectrogram(aw, n_fft, win_len, hop_len, pad)
                spects[n_fft] = spect
            spect_windows.append(spects)

        # Explicitly save spectrogram window lengths (for different stft param tuples)
        window_lengths = {}
        for n_fft, spect in spects.items():
            window_lengths[n_fft] = spect.size()[1]

        return spect_windows, window_lengths

    # https://pytorch.org/audio/stable/_modules/torchaudio/functional.html#spectrogram
    # We make three meaningful modifications. All commented below.
    def spectrogram(
        self, waveform, n_fft, win_length, hop_length, pad,
    ):
        if self.freq_cutoffs:
            waveform = waveform.numpy()
            waveform = torch.from_numpy(
                self.butter_bandpass_filter(waveform, self.freq_cutoffs),
            )

        if pad > 0:
            waveform = pad_waveform(waveform, (pad, pad), "constant")

        window = hann_window(win_length)

        shape = waveform.size()
        waveform = waveform.reshape(-1, shape[-1])

        # https://pytorch.org/docs/stable/generated/torch.stft.html?highlight=stft#torch.stft
        # Modified default padding behavior. Time bins begin t x hop-length as opposed
        # to being centered about t x hop-length.
        spec_f = stft(
            waveform,
            n_fft,
            hop_length,
            win_length,
            window,
            False,
            "reflect",
            False,
            True,
        )

        spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-3:])

        if self.normalize_spects:
            spec_f /= window.pow(2.0).sum().sqrt()
        if self.stft_power is not None:
            spec_f = spec_f.pow(2.0).sum(-1).pow(0.5 * self.stft_power)

        spec_f = torch.ones(spec_f.size()) * 300000 + spec_f
        spec_f = spec_f.log10()

        # Drop rows outside of okayed frequency range
        if self.freq_cutoffs:
            bin_width = self.samp_freq / n_fft
            rows_to_keep = []
            for i in range(n_fft // 2):
                bin_freq = bin_width * i
                if (
                    bin_freq >= self.freq_cutoffs[0]
                    and bin_freq <= self.freq_cutoffs[1]
                ):
                    rows_to_keep.append(i)
            spec_f = spec_f[rows_to_keep, :]

        # Drop last row and column of spectrogram to maintain fixed pixel quantity
        spec_f = spec_f[:-1, :-1]

        return spec_f.float()

    # Write list of spectrogram windows and corresponding (start, stop) times to disk
    # Each window and corresponding time tuple written to distinct file
    def windows_to_disk(self, filename, spect_windows, times):
        main_file_path = self.windowed_spects_dir_path + filename

        now = dt.now()
        datestamp = "{}-{}-{}".format(now.month, now.day, now.year)

        for i, spect_window_dict in enumerate(spect_windows):

            # CAUTION: any changes to schema below may be breaking
            record = {
                "date_generated": datestamp,
                "snr": self.snr,
                "times": times[i],
                "windows": spect_window_dict,
            }

            # If we duplicated windows with noise suffix filename with "window"
            # and "noise" ID numbers.
            if self.snr != -1 and self.num_window_copies > 1:
                window_id = i // self.num_window_copies
                noise_id = i % self.num_window_copies

                # CAUTION: changes to naming convention may be breaking
                file_path = (
                    main_file_path
                    + ".window"
                    + str(window_id)
                    + ".noise"
                    + str(noise_id)
                    + self.spect_file_fmt
                )
            # Otherwise just "window" ID number
            else:
                # CAUTION: changes to naming convention may be breaking
                file_path = main_file_path + ".window" + str(i) + self.spect_file_fmt

            with open(file_path, "wb") as handle:
                pickle.dump(record, handle)

    # Write uncut spect as 2d tensor, 3d batched windows and corresponding
    # song duration to disk
    def spects_to_disk(self, filename, uncut_spects, batched_windows, duration):
        now = dt.now()
        datestamp = "{}-{}-{}".format(now.month, now.day, now.year)

        # CAUTION: any changes to this schema may be breaking
        record = {
            "date": datestamp,
            "snr": self.snr,
            "spects": uncut_spects,
            "batched_windows": batched_windows,
            "duration": duration,
        }

        # CAUTION: changes to naming convention may be breaking
        file_path = self.uncut_spects_dir_path + filename + self.spect_file_fmt

        with open(file_path, "wb") as handle:
            pickle.dump(record, handle)

    # Adds noise signal to each audio window in provided list
    def add_noise(self, windows):
        noisy_windows = []
        for window in windows:
            window_rms = self.rms(window)
            noise_rms = torch.sqrt(window_rms ** 2 / pow(10, self.snr / 10))
            noise = normal(0, noise_rms, window.numpy().shape)
            noisy_window = window + torch.from_numpy(noise)
            noisy_windows.append(noisy_window)
        if len(noisy_windows) > 1:
            return noisy_windows
        else:
            return noisy_windows[0]

    # Load audio waveform
    def load_waveform(self, audio_file_path):
        try:
            cbin = load_cbin(audio_file_path)[0]
        except:
            cbin = np.load(audio_file_path)
        waveform = torch.from_numpy(cbin.astype(np.float32))
        return waveform

    # Root mean squared
    def rms(self, sig):
        return torch.sqrt(torch.mean(sig ** 2))

    # Pad audio s.t. total number of samples is divisible by modulus
    def pad_audio(self, audio_waveform, modulus):
        total_num_samples = len(audio_waveform)
        rem = total_num_samples % modulus
        pad = modulus - rem
        return pad_waveform(audio_waveform, (0, pad), "constant")

    def butter_bandpass(self, lowcut, highcut, order):
        nyq = 0.5 * self.samp_freq
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="bandpass")
        return b, a

    def butter_bandpass_filter(self, data, cutoffs, order=5):
        lowcut, highcut = cutoffs
        b, a = self.butter_bandpass(lowcut, highcut, order)
        y = lfilter(b, a, data)
        return y
