import torch

from data import Corpus
from train import get_dataloader
from generator import beam_search
from model import Seq2SeqTransformer
from utils import load_check_point, load_vocab

start = 1
end = 2
padding = 3
d_model = 512

validate_set = get_dataloader(Corpus(validation_set=True, bidirectional=False), starting_value=start, ending_value=end, padding_value=padding,
                              token_limit=2**10, batch_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2SeqTransformer(padding=padding, device=device, seperate_embedding=True, decople_token_decoder=True, embedding_dim=d_model,
                           num_encoder_layers=8)

load_check_point(model, None, None, "best.pth")
vocab = load_vocab("sp_unigram.vocab")

def detoken(tokens: torch.Tensor | list) -> str:
    tokens = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
    txt = [vocab[i] for i in tokens if i > 3]
    txt = ''.join(txt).replace('▁', ' ').lstrip()
    return txt


with open("validate_translation", "w", buffering=1) as output:
    for pair in validate_set:
        for en, vi, _ in pair:
            en = en.to(device=device).squeeze()
            vi = vi.to(device=device).squeeze()

            print("EN(Ground truth): ", detoken(en), file=output)
            tgt = "NOT ENOUGH MEMORY"
            try:
                tgt = beam_search(model, en, width=4, device=device, limit=1.2, start=start, end=end, pad=padding).tolist()
            except:
                pass

            print("VI(Translate)   : ", detoken(tgt), file=output)
            print("VI(Ground truth): ", detoken(vi), file=output)
            print('-' * 64, file=output)
