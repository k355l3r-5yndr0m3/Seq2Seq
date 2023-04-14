import torch

from torch import nn, optim
from os import listdir, path, remove, mkdir
from fnmatch import fnmatch

def save_check_point(model: nn.Module, optimizer: optim.Optimizer, save_path: str = "checkpoint.pth"):
    state_dict = {
        "model": model.state_dict() if model is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(state_dict, save_path)


def load_check_point(model: nn.Module, optimizer: optim.Optimizer, load_path: str = "checkpoint.pth"):
    state_dict = torch.load(load_path)
    if model is not None:
        model.load_state_dict(state_dict["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer"])


def keep_n_checkpoints(model: nn.Module, optimizer: optim.Optimizer, checkpoints_dir: str = "checkpoints/", keep_n: int = 4):
    if not path.exists(checkpoints_dir):
        mkdir(checkpoints_dir)
    older_checkpoints = [cp for cp in listdir(checkpoints_dir) if fnmatch(cp, "checkpoint-*.pth")]
    newest_idx = max([int(cp.split('-')[1].split('.')[0]) for cp in older_checkpoints]) if len(older_checkpoints) > 0 else None
    if newest_idx is not None:
        if len(older_checkpoints) >= keep_n:
            oldest_idx = min([int(cp.split('-')[1].split('.')[0]) for cp in older_checkpoints])
            remove(path.join(checkpoints_dir, f"checkpoint-{oldest_idx}.pth"))
        save_check_point(model, optimizer, save_path=path.join(checkpoints_dir, f"checkpoint-{newest_idx+1}.pth"))
    else:
        save_check_point(model, optimizer, save_path=path.join(checkpoints_dir, "checkpoint-0.pth"))


def load_latest_checkpoint(model: nn.Module, optimizer: optim.Optimizer, checkpoints_dir: str = "checkpoints/") -> bool:
    if not path.exists(checkpoints_dir):
        return False
    older_checkpoints = [cp for cp in listdir(checkpoints_dir) if fnmatch(cp, "checkpoint-*.pth")]
    newest_idx = max([int(cp.split('-')[1].split('.')[0]) for cp in older_checkpoints]) if len(older_checkpoints) > 0 else None
    if newest_idx is None:
        return False
    load_check_point(model, optimizer, load_path=path.join(checkpoints_dir, f"checkpoint-{newest_idx}.pth"))
    return True


def load_vocab(vocab_file: str) -> list[str]:
    vocab = None
    with open(vocab_file, "r") as vf:
        vocab = vf.readlines()
    vocab = [tok.split()[0] for tok in vocab]
    return vocab


if __name__ == "__main__":
    vocab = load_vocab("./sp_unigram.vocab")
    print(*vocab, sep='\n')

