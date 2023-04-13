import torch

from torch import optim
from torch import nn
from data import Corpus
from train import get_dataloader, train_epoch
from model import Seq2SeqTransformer

start = 1
end = 2
padding = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2SeqTransformer(device=device)
optimizer = optim.Adadelta(model.parameters())
dataloader = get_dataloader(Corpus(), starting_value=start, ending_value=end, padding_value=padding, token_limit=1024)
criterion = nn.CrossEntropyLoss(ignore_index=padding, reduction='sum')

for i in range(10):
    train_epoch(dataloader, model, optimizer, criterion, on_device=device)
