import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from FinalProject.utils import wer, cer


class GreedyDecoder:
    def __init__(self, text_transformer):
        self.text_transformer = text_transformer

    def decode(self, output, labels, label_lengths, blank_label=28, collapse_repeated=True):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        targets = []
        for i, args in enumerate(arg_maxes):
            decode = []
            targets.append(self.text_transformer.int_to_text(labels[i][:label_lengths[i]].tolist()))
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    decode.append(index.item())
            decodes.append(self.text_transformer.int_to_text(decode))
        return decodes, targets


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


# noinspection DuplicatedCode
class SpeechRecognitionModel(nn.Module):
    """Speech Recognition Model Inspired by DeepSpeech 2"""

    def __init__(self, train_data_loader, hparams, n_cnn_layers, n_rnn_layers,
                 rnn_dim, n_class, n_feats, stride=2, dropout=0.1,
                 text_transformer=None, save_model=True):
        super(SpeechRecognitionModel, self).__init__()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 16, 3, stride=stride, padding=3 // 2)  # cnn for extracting heirachal features
        self.cnn.to(self.device)

        # n residual cnn layers with filter size of 16
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(16, 16, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.rescnn_layers.to(self.device)

        self.fully_connected = nn.Linear(n_feats * 16, rnn_dim)
        self.fully_connected.to(self.device)

        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.birnn_layers.to(self.device)

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )
        self.classifier.to(self.device)

        self.train_loader = train_data_loader

        self.hparams = hparams

        self.decoder = GreedyDecoder(text_transformer)

        self.criterion = nn.CTCLoss(blank=28).to(self.device)
        self.optimizer = optim.AdamW(self.parameters(), self.hparams['learning_rate'])
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.hparams['learning_rate'],
                                                       steps_per_epoch=int(len(self.train_loader)),
                                                       epochs=self.hparams['epochs'],
                                                       anneal_strategy='linear')

        self.save_model = save_model

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

    def train_model(self, epoch):
        self.train()
        data_len = len(self.train_loader)
        for batch_idx, _data in enumerate(self.train_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # This seems wrong
            output = self(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = self.criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                           100. * batch_idx / len(self.train_loader),
                    loss.item()))
                if self.save_model:
                    torch.save(self.state_dict(), "./model/asr_model.pt")

            del spectrograms, labels, input_lengths, label_lengths, output, loss

    def test_model(self, test_loader):
        print('\nevaluating...')
        self.eval()
        test_loss = 0
        test_cer, test_wer = [], []
        with torch.no_grad():
            for i, _data in enumerate(test_loader):
                spectrograms, labels, input_lengths, label_lengths = _data
                spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)

                # This seems wrong
                output = self(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1)  # (time, batch, n_class)

                loss = self.criterion(output, labels, input_lengths, label_lengths)
                test_loss += loss.item() / len(test_loader)

                decoded_preds, decoded_targets = self.decoder.decode(output.transpose(0, 1), labels, label_lengths)
                for j in range(len(decoded_preds)):
                    test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                    test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

        avg_cer = sum(test_cer) / len(test_cer)
        avg_wer = sum(test_wer) / len(test_wer)

        print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer,
                                                                                                avg_wer))
