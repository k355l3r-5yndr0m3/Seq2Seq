import torch

from torch import optim
from torch import nn

from train import get_dataloader, train_epoch
from model import Seq2SeqTransformer
from data import Corpus
from utils import keep_n_checkpoints

start = 1
end = 2
padding = 3
epoch_num = 16
num_checkpoints = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2SeqTransformer(device=device)
optimizer = optim.Adadelta(model.parameters())
dataloader = get_dataloader(Corpus(), starting_value=start, ending_value=end, padding_value=padding, token_limit=2**13+2**11)
criterion = nn.CrossEntropyLoss(ignore_index=padding, reduction='sum')

with open("losses_graph", "w") as graph:
    for epoch in range(epoch_num):
        losses = train_epoch(dataloader, model, optimizer, criterion, on_device=device, loss_take_arg=True, return_losses=True)
        print(f"EPOCH {epoch+1}/{epoch_num}: ", end='', file=graph)
        print(*losses, sep=', ', end='\n', file=graph, flush=True)
        keep_n_checkpoints(model, optimizer, keep_n=num_checkpoints)
