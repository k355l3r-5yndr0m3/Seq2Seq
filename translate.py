import torch

from model import Seq2SeqTransformer
from utils import load_check_point, load_vocab
from generator import beam_search

start = 1
end = 2
padding = 3
d_model = 512

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
model = Seq2SeqTransformer(padding=padding, device=device, seperate_embedding=True, decople_token_decoder=True, embedding_dim=d_model,
                           num_encoder_layers=8)

load_check_point(model, None, None, "best.pth")
vocab = load_vocab("sp_unigram.vocab")

while True:
    src = input("EN: ")
    if len(src) == 0:
        break
    # greedy search
    tgt = beam_search(model, src, width=1, device=device, limit=1.8, start=start, end=end, pad=padding).tolist()
    tgt = [vocab[i] for i in tgt if i > 3]
    tgt = ''.join(tgt).replace('▁', ' ').lstrip()
    print(f"VI: {tgt}")
