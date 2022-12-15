from dataset import CommonVoiceDataset
from data_loader import DataLoader, TextTransformer
from model import SpeechRecognitionModel

# dev_dataset = CommonVoiceDataset("C:\\Users\\EricL\\School\\nlp\\Programming Assignments\\FinalProject\\data\\archive\\cv-valid-dev.csv",
#                              "C:\\Users\\EricL\\School\\nlp\\Programming Assignments\\FinalProject\\data\\archive\\cv-valid-dev\\")
train_dataset = CommonVoiceDataset("C:\\Users\\EricL\\School\\nlp\\Programming Assignments\\FinalProject\\data\\archive\\cv-valid-train.csv",
                             "C:\\Users\\EricL\\School\\nlp\\Programming Assignments\\FinalProject\\data\\archive\\cv-valid-train\\")
test_dataset = CommonVoiceDataset("C:\\Users\\EricL\\School\\nlp\\Programming Assignments\\FinalProject\\data\\archive\\cv-valid-test.csv",
                             "C:\\Users\\EricL\\School\\nlp\\Programming Assignments\\FinalProject\\data\\archive\\cv-valid-test\\")

hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 128,
        "n_class": 29,
        "n_feats": 128,
        "stride":2,
        "dropout": 0.1,
        "learning_rate": 5e-4,
        "batch_size": 1,
        "epochs": 10
    }

# dev_loader = DataLoader(dev_dataset, hparams["batch_size"])
train_loader = DataLoader(train_dataset, hparams["batch_size"])
test_loader = DataLoader(test_dataset, hparams["batch_size"])
text_transformer = TextTransformer()

model = SpeechRecognitionModel(train_loader.get_data_loader(), hparams, hparams['n_cnn_layers'], hparams['n_rnn_layers'],
                               hparams['rnn_dim'], hparams['n_class'], hparams['n_feats'], hparams['stride'],
                               hparams['dropout'], text_transformer)

for epoch in range(1, hparams["epochs"] + 1):
    model.train_model(epoch)
model.test_model(test_loader.get_data_loader())

