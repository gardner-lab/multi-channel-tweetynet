"""Simple Helpers"""
from glob import glob
import os
import random
import pickle
import shutil
from evfuncs import readrecf
from src.annotation_analyzer import AnnotationAnalyzer


# Load sample spect window from disk and return dims for all fft parameterizations
def get_spect_window_shapes(windowed_spects_dir_path, spect_file_fmt, n_ffts):
    files = glob(os.path.join(windowed_spects_dir_path, "*" + spect_file_fmt))
    file = random.choice(files)
    with open(file, "rb") as handle:
        spect_record = pickle.load(handle)
    window_sizes = {}
    windows = spect_record["windows"]
    for nfft, window in windows.items():
        if nfft in n_ffts:
            window_sizes[nfft] = (1, 1,) + window.size()
    return window_sizes


def get_dataset_dur(audio_files_list, rec_dir_path):
    rec_files_list = [
        f.replace(".cbin", ".rec") for f in audio_files_list if f.endswith(".cbin")
    ]
    if rec_files_list == []:
        rec_files_list = [
            f.replace(".spect", ".rec")
            for f in audio_files_list
            if f.endswith(".spect")
        ]
    dur = 0
    for f in rec_files_list:
        fpath = os.path.join(rec_dir_path, f)
        rec = readrecf(fpath)
        dur += rec["num_samples"] / rec["sample_freq"]
    return dur


# Make fresh directories from list of dirs
def setup_directories(dir_list):
    for d in dir_list:
        if os.path.isdir(d):
            shutil.rmtree(d)
    for d in dir_list:
        if not os.path.isdir(d):
            os.makedirs(d)


# Extract list of all labels found in provided annotation files
# If 'annot_files' not provided then use all files found in provided directory
# NOTE: this method only support annotations provided in
# https://figshare.com/articles/Bengalese_Finch_song_repository/4805749.
def get_labelset(annot_dir_path, annot_file_fmt, annot_files=None):
    aa = AnnotationAnalyzer(annot_dir_path, annot_file_fmt)
    if annot_files is not None:
        return aa.get_labelset(**{"annot_files": annot_files})
    else:
        return aa.get_labelset()


# Return random sample of training and eval file names
def sample_train_eval_files(
    uncut_spect_path, num_train_files, num_eval_files, audio_file_fmt, spect_file_fmt
):
    all_files = [
        f.split("/")[-1] for f in glob(uncut_spect_path + "*" + spect_file_fmt)
    ]
    tmp = random.sample(all_files, num_train_files)
    eval_files = random.sample(list(set(all_files) - set(tmp)), num_eval_files)
    train_files = [f.replace(spect_file_fmt, audio_file_fmt) for f in tmp]
    return train_files, eval_files
