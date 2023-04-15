import torch

from torch import optim
from torch import nn

from train import get_dataloader, train_epoch
from model import Seq2SeqTransformer
from data import Corpus
from utils import keep_n_checkpoints, load_latest_checkpoint

start = 1
end = 2
padding = 3
epoch_num = 512
num_checkpoints = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2SeqTransformer(device=device)
# optimizer = optim.Adadelta(model.parameters())
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = optim.AdamW(model.parameters())

dataloader = get_dataloader(Corpus(), starting_value=start, ending_value=end, padding_value=padding,
                            token_limit=2**13+2**12+2**10, num_workers=0)
criterion = nn.CrossEntropyLoss(ignore_index=padding, reduction='mean', label_smoothing=0.1)

if load_latest_checkpoint(model, optimizer):
    print("restart from checkpoint.")
else:
    print("training from scratch.")

with open("losses_graph", "w") as graph:
    for epoch in range(epoch_num):
        train_epoch(dataloader, model, optimizer, criterion, on_device=device, loss_take_arg=True, exclaim_each_batch=True)
        print("Saving checkpoint.")
        keep_n_checkpoints(model, optimizer, keep_n=num_checkpoints)
