import torch
import warnings
import sys

from data import Corpus
from typing import Iterable
from torch import nn
from torch import optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchtext.data.functional import sentencepiece_numericalizer, load_sp_model
from torchtext.functional import to_tensor
from model import Seq2SeqTransformer

sp_model = load_sp_model("sp_unigram.model")
sp_unigram = sentencepiece_numericalizer(sp_model)


def collate_fn(tokenizer, starting_value: int, ending_value: int, padding_value: int, token_limit: int = 512):
    tfn = None
    if callable(tokenizer):
        def _tfn(en, vi):
            return (tokenizer(en), tokenizer(vi))
        tfn = _tfn
    else:
        en_tok, vi_tok = tokenizer

        def _tfn(en, vi):
            return en_tok(en), vi_tok(vi)
        tfn = _tfn

    def collate(batch: list[tuple[str, str]]) -> Iterable[tuple[Tensor, Tensor]]:
        en, vi = zip(*batch)
        en, vi = tfn(en, vi)
        pairs = sorted(zip(en, vi), key=lambda pair: max(len(pair[0]), len(pair[1])), reverse=True)
        ntoks = [len(en) + len(vi) for en, vi in pairs]
        batch_size = len(pairs)
        while ntoks[0] * len(ntoks) > token_limit:  # if the produced tensor is too big, split them
            split_size = token_limit // ntoks[0]
            if split_size > 0:
                en, vi = zip(*pairs[:split_size])
                en = to_tensor([[starting_value]+s+[ending_value] for s in en], padding_value=padding_value)
                vi = to_tensor([[starting_value]+s+[ending_value] for s in vi], padding_value=padding_value)
                yield (en, vi, split_size / batch_size)
            else:
                warnings.warn(f'token_limit={token_limit} is too small for some samples in the dataset. Consider rasing it.')
            pairs = pairs[split_size:]
            ntoks = ntoks[split_size:]
        factor = len(pairs) / batch_size
        en, vi = zip(*pairs)
        en, vi = to_tensor(list(en), padding_value=padding_value), to_tensor(list(vi), padding_value=padding_value)
        yield (en, vi, factor)
    return collate


def get_dataloader(corpus: Dataset, batch_size: int = 512, shuffle: bool = True, tokenizer = sp_unigram,
                   starting_value: int = 1, ending_value: int = 2, padding_value: int = 3, token_limit: int = 512,
                   num_workers: int = 0):
    return DataLoader(corpus, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_fn(tokenizer, starting_value, ending_value, padding_value, token_limit))


def train_epoch(dataloader: DataLoader, model: nn.Module, optimizer: optim.Optimizer, criterion: callable,
                scheduler: optim.lr_scheduler.LRScheduler | None = None,
                print_loss: bool = True, aggregate_over_nbatch: int = 1, return_losses: bool = False,
                loss_take_arg: bool = False, exclaim_each_batch: bool = True, on_device: str = 'cpu',
                write_loss_to=sys.stdout, write_batch_num=sys.stdout, switch: bool = False) -> None | list[float]:
    model.train()
    loss_scale_factor = 1.0 / aggregate_over_nbatch if loss_take_arg else 1.0
    loss_aggregate = 0.0
    losses = [] if return_losses else None
    for batch_num, batch in enumerate(dataloader):
        if exclaim_each_batch:
            print(f"BATCH:{batch_num+1}/{len(dataloader)}", file=write_batch_num)
        optimizer.zero_grad()
        for src, tgt, factor in batch:
            src = src.to(device=on_device)
            tgt = tgt.to(device=on_device)
            if switch:
                src, tgt = tgt, src

            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            try:
                logits = model(src=src, tgt=tgt_in)
            except Exception as e:
                print(f"Something is wrong. src: {src.shape}, tgt: {tgt.shape}", file=sys.stderr)
                print(e, file=sys.stderr)
                continue
            logits = torch.flatten(logits, 0, 1)
            tgt_out = torch.flatten(tgt_out, 0, 1)
            loss = criterion(logits, tgt_out) * factor
            loss_aggregate += loss.item() * loss_scale_factor
            loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if (batch_num + 1) % aggregate_over_nbatch == 0 and print_loss:
            print(f"BATCH:{batch_num+1}/{len(dataloader)}->{loss_aggregate}",
                  file=write_loss_to)
            if return_losses:
                losses.append(loss_aggregate)
            loss_aggregate = 0.0
        del batch
    return losses


def validate(dataloader: DataLoader, model: nn.Module, criterion: callable, device: str = 'cpu', switch: bool = False) -> float:
    model.eval()
    total = 0.0
    factor = 1.0 / len(dataloader)
    for batch in dataloader:
        for src, tgt, f in batch:
            src = src.to(device=device)
            tgt = tgt.to(device=device)
            if switch:
                src, tgt = tgt, src

            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            try:
                logits = model(src=src, tgt=tgt_in)
            except Exception as e:
                print(f"Something is wrong. src: {src.shape}, tgt: {tgt.shape}", file=sys.stderr)
                print(e, file=sys.stderr)
                continue
            logits = torch.flatten(logits, 0, 1)
            tgt_out = torch.flatten(tgt_out, 0, 1)
            loss = criterion(logits, tgt_out) * factor * f
            total += loss.item()
            del src, tgt, tgt_in, tgt_out, loss, logits
    return total


if __name__ == "__main__":
    model = Seq2SeqTransformer(device='cuda')
    validate_set = get_dataloader(Corpus(validation_set=True), starting_value=1, ending_value=2, padding_value=3,
                                  token_limit=2**13+2**8)
    criterion = nn.CrossEntropyLoss(ignore_index=3, reduction='mean', label_smoothing=0.1)
    print(validate(validate_set, model, criterion, device='cuda'))



