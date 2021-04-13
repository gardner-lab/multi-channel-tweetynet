from glob import glob
from scipy.io import loadmat
import numpy as np

# Utility for analyzing distribution of labels (syllables) in specified sets of songs
class AnnotationAnalyzer:
    def __init__(self, annot_dir_path, annot_file_fmt):
        self.annot_dir_path = annot_dir_path
        self.annot_file_fmt = annot_file_fmt
        self.extract_methods = {
            "labels": self.extract_labels,
            "onsets": self.extract_onsets,
            "offsets": self.extract_offsets,
        }

    def get_labelset(self, **kwargs):
        labelset = set()
        if "annot_files" in kwargs:
            annot_file_paths_list = [
                self.annot_dir_path + f for f in kwargs["annot_files"]
            ]
        else:
            annot_file_paths_list = glob(
                self.annot_dir_path + "*" + self.annot_file_fmt
            )

        for annot_file_path in annot_file_paths_list:
            (labels,) = self.load_annotation(annot_file_path, ["labels"])
            for l in labels:
                labelset.add(l)
        labels = list(labelset)
        labels.sort()
        return labels

    def get_frequency_distribution(self, **kwargs):
        all_labs = []
        if "annot_files" in kwargs:
            annot_file_paths_list = [
                self.annot_dir_path + f for f in kwargs["annot_files"]
            ]
        else:
            annot_file_paths_list = glob(
                self.annot_dir_path + "*" + self.annot_file_fmt
            )
        for annot_file_path in annot_file_paths_list:
            (labels,) = self.load_annotation(annot_file_path, ["labels"])
            for l in labels:
                all_labs.append(l)
        all_labs = np.array(all_labs)
        unique, counts = np.unique(all_labs, return_counts=True)
        normalized_freqs = counts / np.sum(counts)
        return dict((lab, p) for lab, p in list(zip(unique, normalized_freqs)))

    def get_time_distribution(self):
        raise NotImplementedError()

    def load_annotation(self, annot_file_path, keys):
        annot = loadmat(annot_file_path)
        if keys == []:
            return annot
        else:
            ret = ()
            for k in keys:
                method = self.extract_methods[k]
                v = method(annot)
                ret += (v,)
            return ret

    def extract_labels(self, annot):
        labels_list = annot["labels"]
        return [c for c in labels_list[0]]

    def extract_onsets(self, annot):
        raise NotImplementedError()

    def extract_offsets(self, annot):
        raise NotImplementedError()

