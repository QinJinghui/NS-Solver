import torch
from torch.nn import functional
import torch.nn.functional as F
import numpy as np


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()  # 生成一个0-maxlen的1D tensor
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len) # 将seq_range复制batch_size次
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand  # 返回的tensor中只有小于序列长度的位置为1，大于的话为0


def masked_cross_entropy_with_logit(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # print("logits_flat:", logits_flat.cpu().detach().numpy())
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # print(log_probs_flat.cpu().detach().numpy())
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # print("target_flat:", target_flat.cpu().detach().numpy())
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # print("losses_flat:", losses_flat.cpu().detach().numpy())
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss


def masked_cross_entropy_with_logit_with_answer_mask(logits, target, length, answer_mask):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))

    losses = losses * mask.float() * answer_mask.float()
    # print(answer_mask.size())
    # print(length.float().sum())
    length = length.float() * answer_mask.squeeze(1).float()
    # print(length.float().sum())
    if length.float().sum() > 0:
        loss = losses.sum() / length.float().sum()
    else:
        loss = losses.sum()
    # loss = losses.sum() / length.float().sum()
    # mask_length = length.float() * answer_mask.squeeze(1).float()
    # loss = losses.sum() / mask_length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    # print(loss.item())
    return loss


# def batchPGLoss(self, inp, target, reward):
#     batch_size, seq_len = inp.size()
#     inp = inp.permute(1, 0)          # seq_len x batch_size
#     target = target.permute(1, 0)    # seq_len x batch_size
#     h = self.init_hidden(batch_size)
#
#     loss = 0
#     for i in range(seq_len):
#         out, h = self.forward(inp[i], h)
#         # TODO: should h be detached from graph (.detach())?
#         for j in range(batch_size):
#             loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q
#
#     return loss/batch_size


def masked_answer_pg_loss(logits, target, length, answer_rewards):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    # batch_size, max_len, num_classes
    batch_size, max_len, num_classes = logits.size()
    log_probs = functional.log_softmax(logits, dim=2)
    # print(log_probs)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1)).float()
    loss = 0
    for tl in range(max_len):
        for j in range(batch_size):
            # print(log_probs[j][tl][target.data[j][tl]])
            loss += -log_probs[j][tl][target.data[j][tl]]*answer_rewards[j]*mask[j][tl]

    # return loss/batch_size
    return loss / length.float().sum()


def masked_answer_pg_loss2(logits, target, length, answer_rewards):
    discounted_episode_rewards = np.zeros_like(target.cpu().numpy()).astype(float)
    # print(discounted_episode_rewards.shape)
    episode_rewards = np.zeros_like(target.cpu().numpy()).astype(float)
    answer_rewards_np = answer_rewards.cpu().numpy().tolist()
    gamma = 0.99 # discount factor
    # batch_size, max_len, num_classes
    batch_size, max_len, num_classes = logits.size()
    # set episode_rewards
    for b_idx in range(batch_size):
        # print(answer_rewards_np[b_idx])
        # print(length[b_idx] - 1)
        for i in range(length[b_idx] - 1):
            episode_rewards[b_idx][i] = -1.0
        episode_rewards[b_idx][length[b_idx] - 1] = answer_rewards_np[b_idx][0]
    # print(episode_rewards)
    for b_idx in range(batch_size):
        cumulative = 0
        # print(discounted_episode_rewards[b_idx])
        # print(episode_rewards[b_idx])
        for t in reversed(range(length[b_idx])):
            cumulative = cumulative * gamma + episode_rewards[b_idx][t]
            discounted_episode_rewards[b_idx][t] = cumulative
        discounted_episode_rewards[b_idx] -= np.mean(discounted_episode_rewards[b_idx][0:length[b_idx]])
        discounted_episode_rewards[b_idx] /= np.std(discounted_episode_rewards[b_idx][0:length[b_idx]]) + 1
        # print(discounted_episode_rewards[b_idx])

    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)

    log_probs = functional.log_softmax(logits, dim=2)
    # print(log_probs)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1)).float()
    loss = 0
    for tl in range(max_len):
        for j in range(batch_size):
            # print(log_probs[j][tl][target.data[j][tl]])
            loss += -log_probs[j][tl][target.data[j][tl]]*discounted_episode_rewards[j][tl]*mask[j][tl]

    # return loss / batch_size
    return loss / length.float().sum()


def masked_cross_entropy_without_logit(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))

    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = torch.log(logits_flat + 1e-12)

    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())

    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    if loss.item() > 10:
        print(losses, target)
    return loss


def masked_cross_entropy_with_smoothing_control(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        # loss = loss.masked_select(non_pad_mask).sum()  # average later
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='mean') #reduction='sum')
    return loss