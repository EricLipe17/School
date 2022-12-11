import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torch.utils.data as data


class TextTransformer:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')


class DataPreprocessor:
    def __init__(self, tsv_path, tsv_name, batch_size, data_type="train"):
        self.tsv_path = tsv_path
        self.tsv_name = tsv_name
        self.batch_size = batch_size

        self.data_type = data_type
        if data_type == "train":
            self.audio_transformer = nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                torchaudio.transforms.TimeMasking(time_mask_param=100)
            )
        else:
            self.audio_transformer = torchaudio.transforms.MelSpectrogram()

        self.text_transformer = TextTransformer()
        self.loader = None

    def __process(self, data):
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        for (waveform, _, utterance, _, _, _) in data:
            spec = self.audio_transformer(waveform).squeeze(0).transpose(0, 1)
            spectrograms.append(spec)
            label = torch.Tensor(self.text_transformer.text_to_int(utterance.lower()))
            labels.append(label)
            input_lengths.append(spec.shape[0] // 2)
            label_lengths.append(len(label))

        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms, labels, input_lengths, label_lengths

    def create_data_loader(self):
        dataset = torchaudio.datasets.COMMONVOICE(root=self.tsv_path, tsv=self.tsv_name)

        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.loader = data.DataLoader(dataset=dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      collate_fn=lambda x: self.__process(x),
                                      **kwargs)
