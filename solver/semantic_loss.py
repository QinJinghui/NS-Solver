import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
class SemanticLoss(nn.Module):
    def __init__(self, weigted):
        super(SemanticLoss, self).__init__()
        self.weigted = weigted

    def _sequence_mask(self, classes, target_idx):
        bs = target_idx.size(0)
        if USE_CUDA:
            zero_mask = torch.zeros(bs, classes).cuda()
            one_mask = torch.ones(bs, classes).cuda()
        else:
            zero_mask = torch.zeros(bs, classes)
            one_mask = torch.ones(bs, classes)
        for idx, tar in enumerate(target_idx):
            zero_mask[idx][tar] = 1
            one_mask[idx][tar] = 0
        return zero_mask, one_mask

    def forward(self, pred, tar):
        bs, seq, classes = pred.size()
        tar_flat = tar.view(-1, 1)
        pred_flat = pred.view(-1, classes)
        pred_flat = F.softmax(pred_flat, dim=1)
        zero_mask, one_mask = self._sequence_mask(classes, tar_flat)
        pred_flat = (1 - pred_flat) * one_mask + pred_flat * zero_mask
        loss = torch.mean(torch.prod(pred_flat, 1))
        return self.weigted * torch.log(loss + 1e-12)