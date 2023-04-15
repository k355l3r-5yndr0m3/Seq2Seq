import torch
import math

from torch import nn
from torch import Tensor
from torch.nn import functional as F

class PositionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 512, device=torch.device('cpu')):
        super().__init__()
        frequences = torch.linspace(2.0 / embedding_dim, float(embedding_dim), embedding_dim // 2)
        frequences = 10000 ** frequences
        frequences = 1.0 / frequences
        frequences = frequences.unsqueeze(dim=0)
        frequences = frequences.to(device=device)

        self.register_buffer('frequences', frequences)
        self.embeddind_dim = embedding_dim

    def forward(self, sequence: Tensor) -> Tensor:
        seqlen = sequence.size(dim=-2)
        pos = torch.arange(0, seqlen, 1, dtype=sequence.dtype, device=self.frequences.device).unsqueeze(dim=1)
        pos = pos @ self.frequences
        emb = torch.empty((seqlen, self.embeddind_dim), dtype=sequence.dtype, device=self.frequences.device)
        emb[:, 0::2] = torch.sin(pos)
        emb[:, 1::2] = torch.cos(pos)
        return sequence + emb


class Seq2SeqTransformer(nn.Module):
    def __init__(self, nhead: int = 8, embedding_dim: int = 512, num_encoder_layers: int = 8, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, vocab_size: int = 2**14, dropout: float = 0.1, padding: None | int = None,
                 use_tgt_mask: bool = True, device=torch.device('cpu'), decople_token_decoder: bool = False):
        super().__init__()
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding, device=device)
        self.pos_emb = PositionEmbedding(embedding_dim=embedding_dim, device=device)
        self.dropout = nn.Dropout(p=dropout)
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward, dropout=dropout,
                                          activation=F.leaky_relu, batch_first=True,
                                          device=device)
        self.padding = padding
        self.use_tgt_mask = use_tgt_mask
        self.decople_token_decoder = decople_token_decoder
        if decople_token_decoder:
            self.tok_decoder = nn.Linear(embedding_dim, vocab_size)


    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        tgt_seq_len = tgt.size(dim=-1)

        src_padding_mask = torch.zeros_like(src).float().masked_fill(src == self.padding, float('-inf')) if self.padding is not None else None
        tgt_padding_mask = torch.zeros_like(tgt).float().masked_fill(tgt == self.padding, float('-inf')) if self.padding is not None else None

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=tgt.device) if self.use_tgt_mask else None

        src = self.tok_emb(src)
        src = self.dropout(src)
        src = self.pos_emb(src)

        tgt = self.tok_emb(tgt)
        tgt = self.dropout(tgt)
        tgt = self.pos_emb(tgt)

        logits = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        if self.decople_token_decoder:
            logits = self.tok_decoder(logits)
        else:
            # out = F.normalize(out, dim=-1)
            # rem = F.normalize(self.tok_emb.weight, dim=-1)
            rem = self.tok_emb.weight

            # cosine similarity
            logits = logits @ rem.transpose(0, 1) / math.sqrt(logits.shape[-1])
        return logits




if __name__ == "__main__":  # Testing
    model = Seq2SeqTransformer()
    src = torch.randint(0, 2**14, (4, 10,))
    tgt = torch.randint(0, 2**14, (4, 5,))

    logits = model(src, tgt)
    print(logits.size())
    print(logits)





