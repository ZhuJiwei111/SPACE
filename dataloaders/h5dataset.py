import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from Bio import SeqIO

class GEPDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        self.h5_file = h5py.File(self.file_path, 'r')
        self.sequences = self.h5_file['sequences']
        self.targets = self.h5_file['targets']

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        sequence = self.sequences[idx]
        # bool -> float32
        sequence = sequence.astype(np.float32)
        target = self.targets[idx]

        return {
            'x': sequence,
            'labels': target,
            # "head": "human",
            # "target_length": 896,
        }
    
    def close(self):
        self.h5_file.close()

class GEPBedDataset(Dataset):
    def __init__(self, file_path, bed_path, genome_path, shift_aug, rc_aug, seqlen=196608, dataset_path="/home/jiwei_zhu/disk/Enformer/Data/data"):
        self.file_path = file_path
        self.bed_path = bed_path

        self.h5_file = h5py.File(self.file_path, 'r')
        self.targets = self.h5_file['targets']
        self.bed_file = pd.read_csv(bed_path, sep='\t')
        assert len(self.bed_file) == len(self.targets)

        self.seqlen = seqlen
        self.target_len, self.target_dim = self.targets.shape[1:]

        self.shift_aug = shift_aug
        self.rc_aug = rc_aug

        self.preprocess_data_path = f"{dataset_path}/{file_path[-14:-3]}_196608_{self.shift_aug}_{self.rc_aug}.bin"
        self.start = (196608 - seqlen) // 2
        self.end = self.start + seqlen
        if not os.path.isfile(self.preprocess_data_path):
            self.genome_dict = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))
            self.chrom_length = {chrom: len(self.genome_dict[chrom]) for chrom in self.genome_dict}
            self.preprocess_data()
        print(f"load preprocess data: {self.preprocess_data_path}")

        self.chunk_size_x = 196608 * 4 * 4
        self.chunk_size_label = self.target_len * self.target_dim * 4

    def resize_interval(self, chrom, start, end):
        mid_point = (start + end) // 2
        extend_start = mid_point - self.seqlen // 2
        extend_end = mid_point + self.seqlen // 2
        trimmed_start = max(0, extend_start)
        left_pad = trimmed_start - extend_start
        trimmed_end = min(self.chrom_length[chrom], extend_end)
        right_pad = extend_end - trimmed_end
        return trimmed_start, trimmed_end, left_pad, right_pad

    def get_sequence(self, chrom, start, end):
        trimmed_start, trimmed_end, left_pad, right_pad = self.resize_interval(chrom, start, end)
        sequence = str(self.genome_dict[chrom].seq[trimmed_start:trimmed_end]).upper()
        left_pad_seq = 'N' * left_pad
        right_pad_seq = 'N' * right_pad
        sequence = left_pad_seq + sequence + right_pad_seq
        return sequence

    def sequence_to_onehot(self, sequence):
        mapping = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'N': [0, 0, 0, 0],
        }
        onehot = np.array([mapping[base] for base in sequence], dtype=np.float32)
        return onehot

    def __len__(self):
        return len(self.bed_file)

    def process(self, idx):
        target = self.targets[idx]
        chrom, start, end = self.bed_file.loc[idx, ['chrom', 'start', 'end']]

        if self.shift_aug:
            shift = np.random.randint(-3, 4)
            start += shift
            end += shift

        sequence = self.get_sequence(chrom, start, end)
        onehot = self.sequence_to_onehot(sequence)

        if self.rc_aug:
            if np.random.rand() < 0.5:
                onehot = onehot[::-1, ::-1]
                target = target[::-1]

        return {
            'x': onehot,
            'labels': target,
        }

    def preprocess_data(self):
        print(f"There is no preprocessed data at {self.preprocess_data_path}, so we will preprocess it now.")
        print("Preprocessing data...")
        with open(self.preprocess_data_path, 'wb') as f:
            for idx in tqdm(range(len(self))):
                item = self.process(idx)
                x = item['x']
                labels = item['labels']

                # Write the data
                f.write(x.tobytes())
                f.write(labels.tobytes())

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")
        with open(self.preprocess_data_path, 'rb') as file_handle:
            file_handle.seek(idx * (self.chunk_size_x + self.chunk_size_label))

            x_bytes = file_handle.read(self.chunk_size_x)
            label_bytes = file_handle.read(self.chunk_size_label)

            if len(x_bytes) != self.chunk_size_x or len(label_bytes) != self.chunk_size_label:
                raise ValueError("Incomplete data")

            x_data = np.frombuffer(x_bytes, dtype=np.float32).reshape(196608, 4)
            label_data = np.frombuffer(label_bytes, dtype=np.float32).reshape(self.target_len, self.target_dim)

            return {
                'x': x_data[self.start: self.end],
                "labels": label_data,
            }

    def __iter__(self):
        for idx in range(len(self)):
            data = self[idx]
            if data is not None:
                yield data

    def close(self):
        self.h5_file.close()


class MultiSpeciesDataset(Dataset):
    def __init__(self, file_paths, bed_paths, genome_paths, shift_aug, rc_aug, seqlen=196608, dataset_path="/home/jiwei_zhu/disk/Enformer/Data/data"):
        human_file_path, mouse_file_path = file_paths
        human_bed_path, mouse_bed_path = bed_paths
        human_genome_path, mouse_genome_path = genome_paths
        self.human_dataset = GEPBedDataset(human_file_path, human_bed_path, human_genome_path, shift_aug, rc_aug, seqlen, dataset_path)
        self.mouse_dataset = GEPBedDataset(mouse_file_path, mouse_bed_path, mouse_genome_path, shift_aug, rc_aug, seqlen, dataset_path)
        
        len_human = len(self.human_dataset)
        len_mouse = len(self.mouse_dataset)
        
        if len_human > len_mouse:
            self.larger_species = 'human'
            self.larger_dataset = self.human_dataset
            self.smaller_dataset = self.mouse_dataset
            self.len_larger = len_human
            self.len_smaller = len_mouse
        else:
            self.larger_species = 'mouse'
            self.larger_dataset = self.mouse_dataset
            self.smaller_dataset = self.human_dataset
            self.len_larger = len_mouse
            self.len_smaller = len_human
            
        repeats = self.len_larger // self.len_smaller
        remainder = self.len_larger % self.len_smaller
        #! 对更少的部分随机选取
        self.smaller_indices = np.concatenate([np.arange(self.len_smaller)] * repeats + [np.random.choice(self.len_smaller, remainder, replace=False)])
        
    def __len__(self):
        return self.len_larger
        
    def __getitem__(self, idx):
        larger_data = self.larger_dataset[idx]
        smaller_idx = self.smaller_indices[idx]
        smaller_data = self.smaller_dataset[smaller_idx]
        return {
            'human_x': larger_data['x'] if isinstance(self.larger_dataset, GEPBedDataset) and self.larger_species == 'human' else smaller_data['x'],
            'human_labels': larger_data['labels'] if isinstance(self.larger_dataset, GEPBedDataset) and self.larger_species == 'human' else smaller_data['labels'],
            'mouse_x': larger_data['x'] if isinstance(self.larger_dataset, GEPBedDataset) and self.larger_species == 'mouse' else smaller_data['x'],
            'mouse_labels': larger_data['labels'] if isinstance(self.larger_dataset, GEPBedDataset) and self.larger_species == 'mouse' else smaller_data['labels'],
        }

class VCFDataset(Dataset):
    def __init__(self, file_path, genome_dict, seqlen):
        self.file_path = file_path
        self.vcf_file = pd.read_csv(file_path, sep='\t', header=None, comment='#')
        self.vcf_file.columns = ['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter']
        self.seqlen = seqlen
        self.genome_dict = genome_dict
        self.chrom_length = {chrom: len(genome_dict[chrom]) for chrom in genome_dict}

    def resize_interval(self, chrom, start, end):
        mid_point = (start + end) // 2
        extend_start = mid_point - self.seqlen // 2
        extend_end = mid_point + self.seqlen // 2
        trimmed_start = max(0, extend_start)
        left_pad = trimmed_start - extend_start
        trimmed_end = min(self.chrom_length[chrom], extend_end)
        right_pad = extend_end - trimmed_end
        return trimmed_start, trimmed_end, left_pad, right_pad

    def get_sequence(self, chrom, start, end):
        trimmed_start, trimmed_end, left_pad, right_pad = self.resize_interval(chrom, start, end)
        sequence = str(self.genome_dict[chrom].seq[trimmed_start:trimmed_end]).upper()
        left_pad_seq = 'N' * left_pad
        right_pad_seq = 'N' * right_pad
        sequence = left_pad_seq + sequence + right_pad_seq
        return sequence

    def sequence_to_onehot(self, sequence):
        mapping = {'A': [1, 0, 0, 0],
                   'C': [0, 1, 0, 0],
                   'G': [0, 0, 1, 0],
                   'T': [0, 0, 0, 1],
                   'N': [0, 0, 0, 0]}
        onehot = np.array([mapping[base] for base in sequence], dtype=np.float32)
        return onehot
    
    def __len__(self):
        return len(self.vcf_file)

    def __getitem__(self, idx):
        chrom, pos, ref, alt = self.vcf_file.loc[idx, ['chrom', 'pos', 'ref', 'alt']]
        pos = pos - 1 # in vcf, 1-based coordinate
        ref_sequence = self.get_sequence(chrom, pos, pos + 1)
        # assert the middle base is ref
        assert ref_sequence[self.seqlen // 2] == ref
        # replace the middle base with alt
        alt_sequence = ref_sequence[:self.seqlen // 2] + alt + ref_sequence[self.seqlen // 2 + 1:]
        
        ref_onehot = self.sequence_to_onehot(ref_sequence)
        alt_onehot = self.sequence_to_onehot(alt_sequence)

        return {
            "ref_x": ref_onehot,
            "alt_x": alt_onehot,
        }
