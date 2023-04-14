import torch

from torch import nn, optim
from os import listdir, path, remove, mkdir
from fnmatch import fnmatch

def save_check_point(model: nn.Module, optimizer: optim.Optimizer, save_path: str = "checkpoint.pth"):
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state_dict, save_path)


def load_check_point(model: nn.Module, optimizer: optim.Optimizer, load_path: str = "checkpoint.pth"):
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict["model"])
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

if __name__ == "__main__":
    m = nn.Linear(10, 10)
    o = optim.SGD(m.parameters(), lr=1.0)
    for i in range(6):
        keep_n_checkpoints(m, o)

