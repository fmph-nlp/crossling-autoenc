from torch import nn

class BilingualAutoencoder(nn.Module):
    def __init__(self, n_vocab1, n_vocab2, n_hidden):
        super(BilingualAutoencoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(n_vocab1, n_hidden).double(),
            nn.Sigmoid())
        self.encoder2 = nn.Sequential(
            nn.Linear(n_vocab2, n_hidden).double(),
            nn.Sigmoid())
        self.decoder1 = nn.Sequential(
            nn.Linear(n_hidden, n_vocab1).double(),
            nn.Sigmoid())
        self.decoder2 = nn.Sequential(
            nn.Linear(n_hidden, n_vocab2).double(),
            nn.Sigmoid())

    def forward(self, x1, x2):
        x1 = self.encoder1(x1)
        x1 = self.decoder1(x1)
        x2 = self.encoder2(x2)
        x2 = self.decoder2(x2)
        return x1, x2
