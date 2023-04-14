import torch

from torch import Tensor, nn

from train import sp_unigram
from model import Seq2SeqTransformer
from utils import load_vocab

def beam_search(model: nn.Module, src_text: str, width: int = 8, keep_score: bool = False,
                start: int = 1, end: int = 2, pad: int = 3, tokenizer: callable = sp_unigram,
                device=torch.device('cpu'), limit: int = 64, repeat_penalty: float = float('-inf'),
                temp: float = 0.0) -> list[list[int]] | list[tuple[list[int], float]]:
    model.eval()
    src = torch.tensor([[start]+s+[end] for s in sp_unigram([src_text])][0], device=device)
    candidates: list[tuple(Tensor, float)] = [(torch.tensor([start], device=device), 0.0)]
    while True:
        best_idx = -1
        for i in range(len(candidates)):
            if candidates[i][0][-1] != end and candidates[i][0].shape[0] < limit:
                best_idx = i
                break
        if best_idx == -1:
            break
        best = candidates[best_idx]
        candidates = candidates[0:best_idx] + candidates[best_idx+1:-1]
        logits = model(src, best[0])[-1]
        logits = logits + torch.normal(torch.zeros_like(logits), temp)
        logits[best[0][-1]] += repeat_penalty
        logits = torch.nn.functional.softmax(logits, dim=-1)

        selection = torch.topk(logits, width)
        candidates += [(torch.cat((best[0], token.reshape(1))), best[1] * score.item()) for score, token in zip(selection.values, selection.indices)]
        candidates = sorted(candidates, key=lambda s: s[1], reverse=True)[0:width]
    if keep_score:
        return candidates
    else:
        return [s for s, _ in candidates]



if __name__ == "__main__":
    vocab = load_vocab("sp_unigram.vocab")
    model = Seq2SeqTransformer()
    src = "this is a test."
    res = beam_search(model, src, width=8, temp=0.05)
    res = [[vocab[i] for i in s] for s in res]
    print(*res, sep='\n')







