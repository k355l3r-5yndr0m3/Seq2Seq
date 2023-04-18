import torch
import sys

from torch import Tensor, nn
from torch.nn import functional as F

from train import sp_unigram
from model import Seq2SeqTransformer
from utils import load_vocab, load_latest_checkpoint

from torchtext.functional import to_tensor
from itertools import product


def beam_search(model: nn.Module, src: str | Tensor, width: int = 8, return_score: bool = False,
                start: int = 1, end: int = 2, pad: int = 3, tokenizer: callable = sp_unigram,
                device: str = 'cpu', limit: int = 16, temp: float = 0.0) -> Tensor | tuple[Tensor, Tensor]:
    model.eval()
    src = to_tensor([[start]+s+[end] for s in tokenizer([src])]).to(device=device) if isinstance(src, str) else src
    tgt = to_tensor([[start]]).to(device=device)
    scores = torch.tensor([[0.0]], device=device)
    while True:
        nonterminal = torch.logical_and(tgt[:, -1] != end, tgt[:, -1] != pad).unsqueeze(1)
        if tgt.shape[1] > limit or not torch.any(nonterminal):
            break
        logits = model(src.expand(tgt.shape[0], -1), tgt)[:, -1]
        # Find terminated sentence to ignore
        padding_logits = torch.empty_like(logits).fill_(float('-inf')).masked_fill_(nonterminal, 0.0)
        padding_logits[:, pad] = 0.0

        logits = torch.normal(logits, temp) + padding_logits + scores

        topk = torch.topk(logits, k=width)
        topk_tokens, topk_scores = topk.indices, topk.values
        topk_tokens, topk_scores = torch.flatten(topk_tokens), torch.flatten(topk_scores)

        topk = torch.topk(topk_scores, width)
        topk_tokens, topk_scores, topk_sentences = topk_tokens[topk.indices], topk.values, topk.indices // width
        topk_sentences = tgt[topk_sentences]
        tgt = torch.cat([topk_sentences, topk_tokens.unsqueeze(1)], dim=1)
        scores = topk_scores.unsqueeze(1)
    if return_score:
        return tgt.squeeze(dim=0), scores.squeeze(dim=1)
    else:
        return tgt.squeeze(dim=0)


def contrastive_search(model: nn.Module, embedding: Tensor, src: str | Tensor, k: int = 64, alpha: float = 0.5,
                       start: int = 1, end: int = 2, pad: int = 3, tokenizer: callable = sp_unigram,
                       device: str = 'cpu', limit: int = 16, temp: float = 0.0) -> Tensor:
    model.eval()
    src = torch.tensor([[start]+s+[end] for s in tokenizer([src])][0]).to(device=device) if isinstance(src, str) else src
    tgt = torch.tensor([start]).to(device=device)
    net = F.normalize(embedding, dim=-1).transpose(1, 0)
    while True:
        logits = torch.normal(model(src, tgt)[-1], temp)
        topk = torch.topk(logits, k)
        topk_tokens, topk_logits = topk.indices, topk.values
        topk_token_embs = F.normalize(embedding[topk_tokens], dim=-1)
        similarity = topk_token_embs @ net[:, tgt]
        degeneration_penalty = torch.max(similarity, dim=-1).values
        topk_logits = topk_logits * (1 - alpha) - degeneration_penalty * alpha
        token = topk_tokens[torch.argmax(topk_logits)]
        tgt = torch.cat((tgt, token.unsqueeze(0)))
        if token.item() == end or tgt.shape[0] > limit:
            break
    return tgt




def testing(model: Seq2SeqTransformer, device: str = 'cpu', nalpha: int = 11, test_cases: list[str] | None = None,
            write_result_to=sys.stdout, vocab: list[str] | None = None):
    vocab = load_vocab("sp_unigram.vocab") if vocab is None else vocab
    alphas: [float] = torch.linspace(0.0, 1.0, nalpha, dtype=torch.float).tolist()
    test_cases = [
        "Try to translate this sentence .",
        "Handl TyPo, ad ther odities.",
    ] if test_cases is None else test_cases
    for src, alpha in product(test_cases, alphas):
        result: Tensor = contrastive_search(model, model.tok_emb.weight, src, device=device, alpha=alpha)
        result: list[int] = result.tolist()
        result: list[str] = [vocab[token] for token in result]
        print(f"alpha={alpha} \"{src}\" -> \"", *result, '"', sep='', end='\n', file=write_result_to)






if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqTransformer(device=device)
    testing(model, device)



