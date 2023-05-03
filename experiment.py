import torch

from torch import optim
from torch import nn

from torch.optim import lr_scheduler

from train import get_dataloader, train_epoch, validate
from model import Seq2SeqTransformer
from data import Corpus
from utils import keep_n_checkpoints, load_latest_checkpoint, save_check_point
from generator import testing

start = 1
end = 2
padding = 3
epoch_num = 128
num_checkpoints = 4

d_model = 1024
warmup_step = 4000
split_max_token = 2**13
batch_size = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2SeqTransformer(padding=padding, device=device, seperate_embedding=True, decople_token_decoder=True, embedding_dim=d_model,
                           num_encoder_layers=8, dropout=0.1)
# optimizer = optim.Adadelta(model.parameters())
optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, verbose=True,
                                  lr_lambda=lambda step_num: (d_model**(-0.5)) * min((step_num+1)**(-1/2), step_num*(warmup_step**(-3/2))))
# optimizer = optim.AdamW(model.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss(ignore_index=padding, reduction='mean', label_smoothing=0.1)

start_epoch_num = load_latest_checkpoint(model, optimizer, scheduler)
if start_epoch_num is not None:
    start_epoch_num += 1
    print("restart from checkpoint.")
else:
    start_epoch_num = 0
    print("training from scratch.")


validate_best = float('inf')
with open("train_loss", "a", buffering=1) as graph, open("test_translation", "a", buffering=1) as translation, open("validate_loss", "a", buffering=1) as vali:
    for epoch in range(start_epoch_num, epoch_num):
        # training
        dataloader = get_dataloader(Corpus(bidirectional=False), starting_value=start, ending_value=end, padding_value=padding,
                                    token_limit=split_max_token, batch_size=batch_size)
        print(f"{'-'*8}{epoch+1}/{epoch_num}{'-'*8}", file=graph)
        train_epoch(dataloader, model, optimizer, criterion, scheduler, on_device=device, loss_take_arg=False, write_loss_to=graph)
        print(f"{'='*32}\n", file=graph)
        del dataloader
        # translate some phrases
        print(f"{'-'*8}{epoch+1}/{epoch_num}{'-'*8}", file=translation)
        testing(model, device=device, write_result_to=translation)
        print(f"{'='*32}\n", file=translation)


        # validate
        validate_set = get_dataloader(Corpus(validation_set=True, bidirectional=False), starting_value=start, ending_value=end, padding_value=padding,
                                      token_limit=split_max_token, batch_size=batch_size)
        keep_n_checkpoints(model, optimizer, scheduler, keep_n=num_checkpoints)
        val = validate(validate_set, model, criterion, device=device)
        print(f'EPOCH:{epoch+1}->{val}', file=vali)
        del validate_set
        if validate_best > val:  # saving best performer
            print("Saving best so far... ", val)
            save_check_point(model, None, None, 'best.pth')
        validate_best = min(validate_best, val)
