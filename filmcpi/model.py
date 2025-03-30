import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """
    def __init__(self, input_dim=256, dim=256, output_dim=512):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

    def forward(self, x, y):
        gamma, beta = torch.split(self.fc(x), self.dim, 1)

        output = gamma * y + beta
        output = self.fc_out(output)
        return F.softplus(output)


class FilmCPI(nn.Module):
    def __init__(self, max_p=1024, p_features=25 + 1, embed_dim=256, hidden_size=256, p_filter1=3,p_filter2=9,dd=2):
        # Protein CNN
        super(FilmCPI, self).__init__()
        self.max_protein = max_p
        self.p_embedding = nn.Embedding(p_features, embed_dim)

        self.protein_encoder = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=hidden_size, kernel_size=p_filter1),
            nn.Softplus(),
            nn.BatchNorm1d(hidden_size),
            
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=p_filter2,dilation=dd),
            nn.Softplus(),
            nn.BatchNorm1d(hidden_size)
        )
        self.max_pool=nn.AdaptiveMaxPool1d(1)

        self.compound_encoder = nn.Sequential(
                nn.Linear(1024, 256),
                nn.Dropout(0.1)
                )

        self.film = FiLM()
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.Softplus(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Linear(128, 1)
            )

    def forward(self, compound, protein):
        # Compound Encoder
        cc = self.compound_encoder(compound)

        # Protein Encoder
        p_emb = self.p_embedding(protein)
        p_emb = p_emb.transpose(2, 1)
        pp = self.max_pool(self.protein_encoder(p_emb)).squeeze()

        # Fusion and Decoder
        x = self.film(cc, pp)
        # x=self.film(pp,cc)                  # FilmCPI_pc
        # x = torch.cat((cc, pp), dim=1)      # FilmCPI_concat
        return self.decoder(x)
