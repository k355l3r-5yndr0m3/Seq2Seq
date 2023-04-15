import torch

from torch import optim
from torch import nn

from train import get_dataloader, train_epoch
from model import Seq2SeqTransformer
from data import Corpus
from utils import keep_n_checkpoints, load_latest_checkpoint
from generator import testing

start = 1
end = 2
padding = 3
epoch_num = 512
num_checkpoints = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2SeqTransformer(device=device)
# optimizer = optim.Adadelta(model.parameters())
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

dataloader = get_dataloader(Corpus(), starting_value=start, ending_value=end, padding_value=padding,
                            token_limit=2**13+2**8, num_workers=0)
criterion = nn.CrossEntropyLoss(ignore_index=padding, reduction='mean', label_smoothing=0.1)

if load_latest_checkpoint(model, optimizer):
    print("restart from checkpoint.")
else:
    print("training from scratch.")

with open("losses_graph", "w", buffering=1) as graph, open("translation_test", "w", buffering=1) as translation:
    for epoch in range(epoch_num):
        train_epoch(dataloader, model, optimizer, criterion, on_device=device, loss_take_arg=True, write_loss_to=graph)
        print(f"{'-'*8}{epoch+1}/{epoch_num}{'-'*8}\n", file=graph)
        testing(model, device=device, write_result_to=translation)
        keep_n_checkpoints(model, optimizer, keep_n=num_checkpoints)
