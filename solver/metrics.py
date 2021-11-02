import torch

#define metric
def binary_accuracy(preds, ys, max_len):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds).squeeze(0).cpu()
    # y_trim = min(max_len, len(ys))
    # if len(ys) > max_len:
    #     ys = ys[:y_trim]
    y = [0] * max_len
    for i in ys:
        if i >= max_len:
            continue
        y[i] = 1
    y = torch.FloatTensor(y).cpu()
    correct = (rounded_preds == y).float()
    # print(len(correct))
    acc = correct.sum() / len(correct)
    return acc


def inclusion_match(preds, ys, max_len):
    rounded_preds = torch.round(preds)
    rounded_preds = rounded_preds.squeeze(0).cpu().detach().numpy()
    y_trim = min(max_len, len(ys))
    # if len(ys) > max_len:
    #     ys = ys[:y_trim]
    y = [0] * max_len
    for i in ys:
        if i >= max_len:
            continue
        y[i] = 1
    y = torch.FloatTensor(y).cpu().detach().numpy()

    for idx, t in enumerate(y):
        if t == 1 and rounded_preds[idx] == 0:
            return 0

    return 1


def inclusion_match_with_addon(preds, ys, max_len, add_on=3):
    rounded_preds = torch.round(preds)
    rounded_preds = rounded_preds.squeeze(0).cpu().detach().numpy()
    # y_trim = min(max_len, len(ys))
    # if len(ys) > max_len:
    #     ys = ys[:y_trim]
    y = [0] * max_len
    for i in ys:
        if i >= max_len:
            continue
        y[i] = 1
    y = torch.FloatTensor(y).cpu().detach().numpy()

    num = rounded_preds.sum() + add_on
    num_min = min(num, max_len)
    topv, topi = preds.topk(int(num_min))
    for i in topi.cpu().detach().numpy():
        rounded_preds[i] = 1

    for idx, t in enumerate(y):
        if t == 1 and rounded_preds[idx] == 0:
            return 0

    return 1


def exact_match(preds, ys, max_len):
    rounded_preds = torch.round(preds)
    rounded_preds = rounded_preds.cpu().squeeze(0).detach().numpy()
    # y_trim = min(max_len, len(ys))
    # if len(ys) > max_len:
    #     ys = ys[:y_trim]
    y = [0] * max_len
    for i in ys:
        y[i] = 1
    y = torch.FloatTensor(y).cpu().detach().numpy()

    # print(rounded_preds)
    # print(y)
    for pred, t in zip(rounded_preds, y):
        if pred != t:
            return 0
    # if rounded_preds.tolist() != y.tolist():
    #     return 0
    return 1
