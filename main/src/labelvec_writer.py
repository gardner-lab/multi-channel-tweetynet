from glob import glob
import os
from scipy.io import loadmat
import pickle
import torch


class LabelVecWriter:
    def __init__(
        self,
        annot_dir_path,
        annot_file_fmt,
        spect_file_fmt,
        labelvec_file_fmt,
        windowed_spects_dir_path,
        windowed_labelvec_dir_path,
        uncut_spects_dir_path,
        uncut_labelvec_dir_path,
        num_samples_in_window,
        num_samples_in_stride,
        samp_freq,
        labelmap,
        no_label_sym,
        labelvec_len,
    ):
        self.annot_dir_path = annot_dir_path
        self.annot_file_fmt = annot_file_fmt
        self.spect_file_fmt = spect_file_fmt
        self.labelvec_file_fmt = labelvec_file_fmt
        self.windowed_spects_dir_path = windowed_spects_dir_path
        self.windowed_labelvec_dir_path = windowed_labelvec_dir_path
        self.uncut_labelvec_dir_path = uncut_labelvec_dir_path
        self.uncut_spects_dir_path = uncut_spects_dir_path
        self.num_samples_in_window = num_samples_in_window
        self.num_samples_in_stride = num_samples_in_stride
        self.samp_freq = samp_freq
        self.labelmap = labelmap
        self.no_label_sym = no_label_sym
        self.labelvec_len = labelvec_len
        self.window_dur = self.num_samples_in_window / self.samp_freq
        self.window_stride_dur = self.num_samples_in_stride / self.samp_freq

    # LabelvecWriter entry point, main method.
    def write(self, annot_files_list=None):

        if annot_files_list is None:
            # Collect all annotatiton files from provided annotation directory
            annot_file_paths = glob(
                os.path.join(self.annot_dir_path, "*" + self.annot_file_fmt)
            )
        else:
            annot_file_paths = [self.annot_dir_path + f for f in annot_files_list]

        # Use each annotation to generate a labelvector for corresponding
        # spectrogram and spectrogram windows
        for afp in annot_file_paths:

            file_name = afp.split("/")[-1].split(self.annot_file_fmt)[0]

            # Load and extract hand-written audio file annotations from
            # which labelvectors will be generated
            annot_file_name = file_name + self.annot_file_fmt
            annot = self.load_annotation(annot_file_name)

            # Load duration of uncut spectrogram corresponding to current annotation
            spect_file_path = (
                self.uncut_spects_dir_path + file_name + self.spect_file_fmt
            )
            spect_duration = self.load_spect_duration(spect_file_path)

            # Complete spect can be batched as individual windows. Need
            # to know how many windows so we know how long the labelvec should be.
            num_unique_windows = self.load_number_unique_windows(spect_file_path)
            uncut_labelvec_len = self.labelvec_len * num_unique_windows

            # Generate and write uncut labelvec
            uncut_labelvec = self.extract_labelvec(
                0, spect_duration, annot, uncut_labelvec_len
            )
            # Reshape uncut labelvec into batches of window sized labelvecs
            batched_labelvec = self.batch_uncut_labelvec(
                uncut_labelvec, num_unique_windows
            )

            # 1D "uncut" tensor and 2D "batched" representation of labelvec to disk
            uncut_labelvec_file_name = file_name + self.labelvec_file_fmt
            self.spect_labelvec_to_disk(
                uncut_labelvec_file_name, uncut_labelvec, batched_labelvec,
            )

            # Collect spectrogram window files corresponding to the current annotation
            spect_window_files = glob(self.windowed_spects_dir_path + file_name + "*")
            for wf in spect_window_files:

                # load precomputed window time bounds
                window_start_time, stop_time = self.load_window_times(wf)

                # Generate window specific labelvec
                window_labelvec = self.extract_labelvec(
                    window_start_time, stop_time, annot, self.labelvec_len
                )

                # Write window labelvec to disk
                window_number = int(wf.split(".window")[1].split(".")[0])
                window_labelvec_file_name = (
                    file_name + ".window" + str(window_number) + self.labelvec_file_fmt
                )
                self.window_labelvec_to_disk(
                    window_labelvec_file_name, window_labelvec,
                )

    def extract_labelvec(self, start, stop, annot, labelvec_len):
        song_dur = stop - start

        # Duration of a single label
        labelbin_dur = song_dur / labelvec_len

        labelvec = []

        # Build labelvector index by index
        for i in range(labelvec_len):

            # Beginning and end of current labelbin
            idx_start = start + (i * labelbin_dur)
            idx_stop = min(idx_start + labelbin_dur, stop)

            # Where to collect labelbin, annotation overlaps
            overlaps = []

            # For each entry in the annotation calculate overlap with labelbin
            for onset, offset, lab in annot:
                overlap_dur = self.calc_overlap_dur(
                    (onset, offset), (idx_start, idx_stop)
                )
                if overlap_dur > 0:
                    overlaps.append((overlap_dur, lab))

            # sort overlaps by duration
            overlaps.sort(key=lambda x: x[0])

            # if there are no overlaps the segment is labeled as silence
            if len(overlaps) == 0:
                label = self.no_label_sym

            # otherwise check if recorded overlap(s) was majority of labelbin dur
            else:
                max_overlap_dur = overlaps[0][0]

                # if not majority segment labeled as silence
                if max_overlap_dur < (labelbin_dur / 2):
                    label = self.no_label_sym

                # otherwise segment labeled appropriately
                else:
                    label = overlaps[0][1]

            integer_label = self.labelmap[label]
            labelvec.append(integer_label)

        return torch.tensor(labelvec)

    def window_labelvec_to_disk(self, file_name, labelvec):
        file_path = self.windowed_labelvec_dir_path + file_name
        with open(file_path, "wb") as handle:
            pickle.dump(labelvec, handle)

    def spect_labelvec_to_disk(self, file_name, labelvec, batched_labelvec):
        file_path = self.uncut_labelvec_dir_path + file_name
        labelvec_record = {
            "complete": labelvec,
            "batched": batched_labelvec,
        }
        with open(file_path, "wb") as handle:
            pickle.dump(labelvec_record, handle)

    def batch_uncut_labelvec(self, labelvec, number_of_windows):
        return labelvec.view(number_of_windows, self.labelvec_len)

    def load_window_times(self, file_path):
        with open(file_path, "rb") as handle:
            window_record = pickle.load(handle)
        return window_record["times"]

    def load_number_unique_windows(self, file_path):
        with open(file_path, "rb") as handle:
            spect_record = pickle.load(handle)
        batched_windows = list(spect_record["batched_windows"].values())
        samp = batched_windows[0]
        number_windows = samp.size()[0]
        return number_windows

    def load_spect_duration(self, file_path):
        with open(file_path, "rb") as handle:
            spect_record = pickle.load(handle)
        return spect_record["duration"]

    def load_annotation(self, file_name):
        annotations = []
        annot_file_path = self.annot_dir_path + file_name
        annot = loadmat(annot_file_path)
        annots = [annot]
        for a in annots:
            onsets = a["onsets"].flatten()
            offsets = a["offsets"].flatten()
            labels = [lab for lab in a["labels"][0]]
            annotations.append(list(zip(onsets, offsets, labels)))
        if len(annotations) == 1:
            return annotations[0]
        else:
            return annotations

    # Calculates overlap duration of two segments (tuples)
    def calc_overlap_dur(self, seg1, seg2):
        return max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
