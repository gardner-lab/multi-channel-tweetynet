from glob import glob
import pickle
from torch.utils.data import Dataset as TorchDataset

# Indexing returns (1) dictionary spectrogram windows derived from provided audio file list
# where each value in dictionary is the same audio window transformed with distinct stft parameters
# and (2) a labelvector corresponding to the window after being downsampled in tweetynet
class TrainDataset(TorchDataset):
    def __init__(
        self,
        spect_files_dir_path,
        labelvec_files_dir_path,
        audio_file_fmt,
        spect_file_fmt,
        labelvec_file_fmt,
        audio_files_list,
        network_nffts,
    ):
        self.spect_files_dir_path = spect_files_dir_path
        self.labelvec_files_dir_path = labelvec_files_dir_path
        self.spect_file_fmt = spect_file_fmt
        self.labelvec_file_fmt = labelvec_file_fmt
        self.network_nffts = network_nffts
        self.audio_file_fmt = audio_file_fmt
        self.window_files_list = self.get_window_files_list(audio_files_list)

    # Load all spectrogram window files corresponding to provided audio files
    def get_window_files_list(self, audio_files_list):
        window_files_list = []
        for f in audio_files_list:
            filename = f.split(self.audio_file_fmt)[0]
            window_files_path = self.spect_files_dir_path + filename + ".*"
            window_files = glob(window_files_path)
            window_files_list += window_files
        return window_files_list

    def get_labelvec_file_path(self, spect_file_path):
        filename = spect_file_path.replace(self.spect_files_dir_path, "").split(
            ".window"
        )[0]
        window_number = spect_file_path.split(".window")[1].split(".")[0]
        return (
            self.labelvec_files_dir_path
            + filename
            + ".window"
            + window_number
            + self.labelvec_file_fmt
        )

    def load(self, file_path):
        with open(file_path, "rb") as handle:
            return pickle.load(handle)

    def __len__(self):
        return len(self.window_files_list)

    def __getitem__(self, idx):
        window_file_path = self.window_files_list[idx]
        labelvec_file_path = self.get_labelvec_file_path(window_file_path)
        window_record = self.load(window_file_path)
        labelvec = self.load(labelvec_file_path)
        windows = window_record["windows"]
        record_nffts = list(windows.keys())
        for nfft in record_nffts:
            if nfft not in self.network_nffts:
                del windows[nfft]
        return windows, labelvec


# Indexing returns (1) a complete spectrogram structured as a minibatch of windows for tweetynet
# to process one at a time and (2) the corresponding labelvector also appropriately batched
class EvalDataset(TorchDataset):
    def __init__(
        self,
        spect_files_dir_path,
        labelvec_files_dir_path,
        spect_files_list,
        spect_file_fmt,
        labelvec_file_fmt,
        network_nffts,
    ):
        self.spect_files_dir_path = spect_files_dir_path
        self.labelvec_files_dir_path = labelvec_files_dir_path
        self.spect_file_fmt = spect_file_fmt
        self.labelvec_file_fmt = labelvec_file_fmt
        self.network_nffts = network_nffts
        self.spect_files_list = [
            self.spect_files_dir_path + f for f in spect_files_list
        ]
        self.labelvec_files_list = self.get_labelvec_files_from(spect_files_list)

    # Load labelvec files corresponding to provided spectrogram files
    def get_labelvec_files_from(self, spect_files_list):
        all_labelvec_files = []
        for f in spect_files_list:
            filename = f.split(self.spect_file_fmt)[0]
            labelvec_file_path = (
                self.labelvec_files_dir_path + filename + self.labelvec_file_fmt
            )
            all_labelvec_files.append(labelvec_file_path)
        return all_labelvec_files

    def load(self, file_path):
        with open(file_path, "rb") as handle:
            return pickle.load(handle)

    def __len__(self):
        return len(self.spect_files_list)

    def __getitem__(self, idx):
        spect_file_path = self.spect_files_list[idx]
        labelvec_file_path = self.labelvec_files_list[idx]
        spect_record = self.load(spect_file_path)
        labelvecs = self.load(labelvec_file_path)
        spects = spect_record["batched_windows"]
        record_nffts = list(spects.keys())
        for nfft in record_nffts:
            if nfft not in self.network_nffts:
                del spects[nfft]
        return spects, labelvecs["batched"]
