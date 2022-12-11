import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio


class CommonVoiceDataset(Dataset):
    def __init__(self, csv_file, data_dir):
        self.dataset_df = pd.read_csv(csv_file)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        voice_file = os.path.join(self.data_dir, self.dataset_df.iloc[idx, 0])
        waveform = torchaudio.load(voice_file)[0]
        text_label = self.dataset_df.iloc[idx, 1]

        sample = {"waveform": waveform, "text_label": text_label}

        return sample
