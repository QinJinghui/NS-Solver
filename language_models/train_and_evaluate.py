import torch
import os
import time
import pickle
import random

import numpy as np
import torch
import torch.nn as nn

from masked_cross_entropy import *

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()
PAD_token = 0


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz, device):
    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(bptt, source, i):
    # get_batch subdivides the source data into chunks of length CFG.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def get_batch_for_train(source):
    seq_len = source.size(0)
    data = source[0:seq_len - 1]
    target = source[1:seq_len]
    return data, target


def train_language_model(input_batch, input_length, language_model,
                language_model_optimizer, clip=0, model_type='Transformer'):
    # 构建序列掩码，需要的位置置为0，pad置为1，因为masked_fill_需要的格式就是和逻辑反过来的
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    new_input_length = []
    for i in input_length:
        new_input_length.append(i-1)
    #
    # # Trun padded arrays into (batch_size x max_len) tensors, transpos into (max_len x batch_size)
    batch_size = len(input_length)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    language_model.train()

    data, target = get_batch_for_train(input_var)
    if USE_CUDA:
        data = data.cuda()
        target = target.cuda()
        seq_mask = seq_mask.cuda()

    # Zero gradients of both optimizers
    language_model_optimizer.zero_grad()

    if model_type != 'Transformer':
        hidden = language_model.init_hidden(batch_size)

    # Run words through encoder
    if model_type == 'Transformer':
        output = language_model(data)
    else:
        # hidden = repackage_hidden(hidden)
        output, hidden = language_model(data, hidden)

    loss = masked_cross_entropy_with_logit(
        output.transpose(0, 1).contiguous(),  # -> batch x seq
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        new_input_length
    )

    loss.backward()
    return_loss = loss.item()

    # Clip gradient norms
    if clip:
        torch.nn.utils.clip_grad_norm_(language_model.parameters(), clip)
        # Update parameters with optimizers
        # for p in language_models.parameters():
        #     p.data.add_(-lr, p.grad.data)

    # Update parameters with optimizers
    language_model_optimizer.step()

    return return_loss


def evaluate_language_model(input_batch, input_length, language_model, model_type='Transformer'):
    language_model.eval()

    new_input_length = []
    for i in input_length:
        new_input_length.append(i-1)


    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    batch_size = input_var.size(1)

    if model_type != 'Transformer':
        hidden = language_model.init_hidden(bsz=batch_size)

    data, target = get_batch_for_train(input_var)
    if USE_CUDA:
        data = data.cuda()
        target = target.cuda()

    with torch.no_grad():
        # for i in range(0, data_source.size(0) - 1, CFG.bptt):
        #     data, targets = get_batch(CFG.bptt, data_source, i)
        if model_type == 'Transformer':
            output = language_model(data)
        else:
            output, hidden = language_model(data, hidden)
            # hidden = repackage_hidden(hidden)

        loss = masked_cross_entropy_with_logit(
            output.transpose(0, 1).contiguous(),  # -> batch x seq
            target.transpose(0, 1).contiguous(),  # -> batch x seq
            new_input_length
        )

        outputs = F.log_softmax(output, dim=-1)
        topv, topi = outputs.topk(1)
        # print(outputs.size())
        # print(topv.size())
        # print(topi)
        # print(topi.size())

        return loss.item(), topi.view(batch_size, -1)
        # output_flat = output.view(-1, language_vocab)
        # total_loss += len(data) * criterion(output_flat, targets).item()

    # return total_loss / len(data_source)
