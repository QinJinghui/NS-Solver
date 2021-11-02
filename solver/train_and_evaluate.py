from masked_cross_entropy import *
import torch
from models import *
import math
import time
from copy import deepcopy
import copy
from semantic_loss import SemanticLoss
import random
import torch.nn.functional as f
import torch.nn as nn
from utils import *

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()
PAD_token = 0


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    # nums_stack记录的是等式中的单词不在outlang中，但在数字列表中的数字
    # 或等式中的单词不在outlang中，且不在数字列表中的特殊数字
    target_input = copy.deepcopy(target) # 用于生成只有操作符的骨架
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    # 遍历目标序列
    for i in range(len(target)):
        if target[i] == unk:
            # 为unk的elem从nums list中选择正确的模板数字
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        # 如果elem为数字模板或unk，则将其置位为0
        if target_input[i] >= num_start:
            target_input[i] = 0
    # 替换了unk符的方程等式，只有操作符的等式骨架
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_tree_input_with_num_mask(target, decoder_output, nums_stack_batch, num_start, num_mask, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    # nums_stack记录的是等式中的单词不在outlang中，但在数字列表中的数字
    # 或等式中的单词不在outlang中，且不在数字列表中的特殊数字
    target_input = copy.deepcopy(target) # 用于生成只有操作符的骨架
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
        num_mask = num_mask.cpu()

    num_mask = num_mask.numpy()

    num_idx_lists = []
    for i in range(len(num_mask)):
        num_idx_list = []
        for idx, num in enumerate(num_mask[i]):
            if num == 0:
                num_idx_list.append(idx)
        num_idx_lists.append(num_idx_list)
    # 遍历目标序列
    for i in range(len(target)):
        if target[i] == unk:
            # 为unk的elem从nums list中选择正确的模板数字
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            # print(num_stack)
            if len(num_stack) > 1:
                target[i] = num_idx_lists[i][num_stack[0]] + num_start
                # print(i, target[i])
            for num in num_stack:
                num_idx = num_idx_lists[i][num]
                if decoder_output[i, num_start + num_idx] > max_score:
                    target[i] = num_idx + num_start
                    max_score = decoder_output[i, num_start + num_idx]
            # print("final:", i, target[i], max_score)
            #         print(max_score)
            # print("final:", max_score)
        # 如果elem为数字模板或unk，则将其置位为0
        if target_input[i] >= num_start:
            target_input[i] = 0
    # 替换了unk符的方程等式，只有操作符的等式骨架
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    # 替换了unk符的方程等式
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos, batch_first=False):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end  # 判断decoder_input是否为数字, 数字为1，非数字为0
    num_mask_encoder = num_mask < 1  # 非数字为1，数字为0
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size  # B x embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    if batch_first:
        all_embedding = encoder_outputs.contiguous()
    else:
        all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start # 用于计算num pos
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):   # 将数字替换为数字在句子中的具体位置，非数字的为0
        indices[k] = num_pos[k][indices[k]]  # 非数字的为0也会提取到第一个数字的位置，这样不会和真正需要第一个位置的冲突？
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)  # decoder_input B x 1
    if batch_first:
        sen_len = encoder_outputs.size(1)
    else:
        sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len  # 记录每个batch在all embedding上的位置
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices # 记录数字在encoder的位置，即文本的位置
    num_encoder = all_embedding.index_select(0, indices) # 提取题目中数字的embedding
    return num_mask, num_encoder, num_mask_encoder


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size, batch_first=False):
    indices = list()
    if batch_first:
        sen_len = encoder_outputs.size(1)
    else:
        sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)  # 用于记录数字在问题的位置
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]  # 屏蔽多余的数字，即词表中不属于该题目的数字
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index) # B x num_size x H
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    if batch_first:
        all_outputs = encoder_outputs.contiguous() # B x S x H
    else:
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()  # S x B x H
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H or B x S x H -> (S x B) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index.bool(), 0.0)  # 屏蔽其他无关数字


def train_ns_solver(input_batch, input_length, target_batch, target_length, dual_target_batch,
                                                    dual_target_length, nums_stack_batch, num_size_batch, generate_nums,
                                                    input_lm_prob_batch, target_lm_prob_batch, s2t_encoder, s2t_predict,
                                                    s2t_generate, s2t_merge, t2s_encoder, t2s_decoder, ssl_num_pred, ssl_num_loc_pred,
                                                    constant_reasoning, s2t_encoder_optimizer, s2t_predict_optimizer,
                                                    s2t_generate_optimizer, s2t_merge_optimizer, t2s_encoder_optimizer,
                                                    t2s_decoder_optimizer,  ssl_num_pred_optimizer, ssl_num_loc_pred_optimizer,
                                                    constant_reasoning_optimizer, output_lang, dual_output_lang,
                                                    ssl_num_pred_loss_fn, ssl_num_loc_pred_loss_fn, constant_reasoning_loss_fn,
                                                    max_problem_len, max_nums_len, constants, num_pos, var_nums=[],
                                                    use_teacher_forcing=1, r1=0.0005, r2=0.01, r3=1.0, r4=1.0, r5=1.0,r6=0.005, r7=0.1):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    target_constants = []
    constant_start = output_lang.num_start
    for constant_list in constants:
        constant_label = [0 for _ in range(len(generate_nums))]
        for constant in constant_list:
            constant_label[constant - constant_start] = 1
        target_constants.append(constant_label)
    target_constants = torch.FloatTensor(target_constants)

    target_pos = []
    for n_pos in num_pos:
        pos_list = [0] * max_problem_len
        for pos in n_pos:
            pos_list[pos] = 1
        target_pos.append(pos_list)
    target_loc = torch.FloatTensor(target_pos)

    dual_seq_mask = []
    max_len = max(target_length)
    for i in target_length:
        dual_seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    dual_seq_mask = torch.ByteTensor(dual_seq_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)
    dual_target = torch.LongTensor(dual_target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(s2t_predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    s2t_encoder.train()
    s2t_predict.train()
    s2t_generate.train()
    s2t_merge.train()
    t2s_encoder.train()
    t2s_decoder.train()

    ssl_num_pred.train()
    ssl_num_loc_pred.train()
    constant_reasoning.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        dual_seq_mask = dual_seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()


    # Zero gradients of both optimizers
    # s2t_encoder_optimizer.zero_grad()
    # s2t_predict_optimizer.zero_grad()
    # s2t_generate_optimizer.zero_grad()
    # s2t_merge_optimizer.zero_grad()
    #
    # ssl_num_pred_optimizer.zero_grad()
    # ssl_num_loc_pred_optimizer.zero_grad()
    # constant_reasoning_optimizer.zero_grad()

    # Run words through encoder

    s2t_encoder_outputs, s2t_problem_output = s2t_encoder(input_var, input_length)

    ssl_num_prediction_logits = ssl_num_pred(s2t_encoder_outputs.transpose(0,1))
    ssl_num_loc_prediction_logits = ssl_num_loc_pred(s2t_encoder_outputs.transpose(0,1))
    constant_prediction_logits = constant_reasoning(s2t_encoder_outputs.transpose(0,1))
    sigmoid = nn.Sigmoid()
    rounded_constant_preds = torch.round(sigmoid(constant_prediction_logits))
    rounded_constant_preds = rounded_constant_preds.cpu().detach().numpy()

    # num = rounded_constant_preds.sum() + 3
    # topv, topi = constant_prediction_logits.topk(int(num))
    # for i in topi:
    #     rounded_constant_preds[i] = 1
    # print(constants)
    num_mask = []
    target_num_size = []
    max_num_size = max(num_size_batch) + len(generate_nums) + len(var_nums) # 最大的位置列表数目+常识数字数目+未知数列表
    for idx, i in enumerate(num_size_batch):
        d = i + len(generate_nums) + len(var_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
        target_num = [0] * max_nums_len
        target_num[i-1] = 1
        target_num_size.append(target_num)

        # for constant in constants[idx]:
        # for id in range(len(generate_nums)):
        #     if id + constant_start not in constants[idx]:
        #         num_mask[-1][id + len(var_nums)] = 1
        # # print(constants[idx], num_mask[-1])
        rounded_constant_pred = rounded_constant_preds[idx]
        # print(input_var[idx])
        # print([in_lang.index2word[t_id] for t_id in input_var.transpose(0, 1)[idx]])
        # print(s2t_encoder_outputs.size())
        # print(s2t_encoder_outputs.transpose(0,1).size())
        # print(s2t_encoder_outputs.transpose(0,1)[idx])
        # print(constant_prediction_logits[idx])
        # print(rounded_constant_pred)
        num = rounded_constant_pred.sum() + 3
        num_min = min(num, len(generate_nums))
        # print(num_min,num, len(generate_nums))
        topv, topi = constant_prediction_logits[idx].topk(int(num_min))
        # print(topi)
        constant_list = topi.cpu().numpy().tolist()
        for constant in constants[idx]:
            constant_list.append(constant - constant_start)
        # print(constant_list)
        for j in range(len(generate_nums)):
            if j not in constant_list:
                num_mask[-1][j + len(var_nums)] = 1
        # print(constants[idx], num_mask[-1])

    num_mask = torch.ByteTensor(num_mask)  # 用于屏蔽无关数字，防止生成错误的Nx
    target_num_size = torch.FloatTensor(target_num_size)

    if USE_CUDA:
        num_mask = num_mask.cuda()

    node_stacks = [[TreeNode(_)] for _ in s2t_problem_output.split(1, dim=0)] # root embedding B x 1

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    # 提取与问题相关的数字embedding
    all_nums_encoder_outputs = get_all_number_encoder_outputs(s2t_encoder_outputs, num_pos, batch_size, num_size,
                                                              s2t_encoder.hidden_size)

    num_start = output_lang.num_start - len(var_nums)
    embeddings_stacks = [[] for _ in range(batch_size)]  # B x 1  当前的tree state/ subtree embedding / output
    left_childs = [None for _ in range(batch_size)]  # B x 1

    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = s2t_predict(
            node_stacks, left_childs, s2t_encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)
        # if 89 in target[t].tolist():
        #     print("Hello")
        target_t, generate_input = generate_tree_input_with_num_mask(target[t].tolist(), outputs, nums_stack_batch, num_start, num_mask, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = s2t_generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            # 未知数当数字处理，SEP当操作符处理
            if i < num_start:  # 非数字
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), terminal=False))
                # print("Non Number:", node_label[idx].unsqueeze(0).size())
            else:  # 数字
                # if i - num_start >= current_nums_embeddings.size(1) or i == len(output_lang.index2word) - 1:
                #     print("Hello")
                # print(idx, i, num_start)
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    # print("Number:", op.embedding.size())
                    # print("Number:", sub_stree.embedding.size())
                    # print("Number:", current_num.size())
                    current_num = s2t_merge(op.embedding, sub_stree.embedding, current_num)  # Subtree embedding
                o.append(TreeEmbedding(current_num, terminal=True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous() # B x S

    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        ssl_num_loc_prediction_logits = ssl_num_loc_prediction_logits.cuda()
        ssl_num_prediction_logits = ssl_num_prediction_logits.cuda()
        target = target.cuda()
        target_num_size = target_num_size.cuda()
        target_loc = target_loc.cuda()
        target_constants = target_constants.cuda()

    # t2s part
    # equation mask
    equation_mask = []
    max_equ_len = max(target_length)
    for i in target_length:
        equation_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_equ_len)])
    equation_mask = torch.ByteTensor(equation_mask)

    t2s_encoder_outputs, t2s_encoder_hidden = t2s_encoder(target, target_length)
    # hidden = t2s_decoder.init_hidden(batch_size)
    # print(len(hidden))
    # print(t2s_encoder_hidden.size())
    # print(hidden[0][1].size())
    hidden = t2s_encoder_hidden.unsqueeze(0) # [t2s_encoder_hidden, hidden[0][1]]
    init_hidden = torch.zeros_like(hidden)
    t2s_decoder_hidden = torch.cat([hidden, init_hidden],dim=0)
    # t2s_decoder_hidden = torch.cat([hidden, hidden],dim=0)
    t2s_decoder_input = torch.LongTensor([dual_output_lang.word2index["SOS"]] * batch_size)
    max_dual_target_length = max(dual_target_length)
    t2s_all_decoder_outputs = torch.zeros(max_dual_target_length, batch_size, t2s_decoder.classes_size)

    if USE_CUDA:
        t2s_all_decoder_outputs = t2s_all_decoder_outputs.cuda()

    if random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_dual_target_length):
            t2s_decoder_input = t2s_decoder_input.unsqueeze(0)
            if USE_CUDA:
                t2s_decoder_input = t2s_decoder_input.cuda()
            else:
                t2s_decoder_input = t2s_decoder_input.clone()
            t2s_decoder_output, t2s_decoder_hidden = t2s_decoder(t2s_decoder_input, t2s_decoder_hidden, t2s_encoder_outputs, dual_seq_mask)
            t2s_all_decoder_outputs[t] = t2s_decoder_output
            t2s_decoder_input = dual_target[t]
    else:
        for t in range(max_dual_target_length):
            t2s_decoder_input = t2s_decoder_input.unsqueeze(0)

            if USE_CUDA:
                t2s_decoder_input = t2s_decoder_input.cuda()
            else:
                t2s_decoder_input = t2s_decoder_input.clone()
            # print(t2s_decoder_input.size())
            # print(t2s_decoder_hidden.size())
            t2s_decoder_output, t2s_decoder_hidden = t2s_decoder(
                t2s_decoder_input, t2s_decoder_hidden, t2s_encoder_outputs, dual_seq_mask)
            t2s_decoder_output = f.log_softmax(t2s_decoder_output, dim=1) # B x classes_size
            t2s_all_decoder_outputs[t] = t2s_decoder_output
            t2s_decoder_input = torch.argmax(t2s_decoder_output, dim=1)
            # print(t2s_decoder_input.size())

    if USE_CUDA:
        target = target.cuda()
        dual_target = dual_target.cuda()


    s2t_loss = masked_cross_entropy_with_logit(all_node_outputs, target, target_length)
    t2s_loss = masked_cross_entropy_with_logit(t2s_all_decoder_outputs.transpose(0,1).contiguous(), dual_target.transpose(0,1).contiguous(), dual_target_length)
    dual_loss = (np.mean(target_lm_prob_batch) - t2s_loss - np.mean(input_lm_prob_batch) + s2t_loss) ** 2
    loss_num_pred = ssl_num_pred_loss_fn(ssl_num_prediction_logits, target_num_size)
    loss_num_loc_pred = ssl_num_loc_pred_loss_fn(ssl_num_loc_prediction_logits, target_loc)
    loss_constant_pred = constant_reasoning_loss_fn(constant_prediction_logits, target_constants)

    semantic_loss_s2t = SemanticLoss(r2)
    semantic_loss_t2s = SemanticLoss(r7)
    if USE_CUDA:
        semantic_loss_s2t = semantic_loss_s2t.cuda()
        semantic_loss_t2s = semantic_loss_t2s.cuda()

    s2t_semantic_loss = semantic_loss_s2t(all_node_outputs, target)
    t2s_semantic_loss = semantic_loss_t2s(t2s_all_decoder_outputs.transpose(0,1).contiguous(), dual_target.transpose(0,1).contiguous())


    s2t_encoder_optimizer.zero_grad()
    s2t_predict_optimizer.zero_grad()
    s2t_generate_optimizer.zero_grad()
    s2t_merge_optimizer.zero_grad()

    ssl_num_pred_optimizer.zero_grad()
    ssl_num_loc_pred_optimizer.zero_grad()
    constant_reasoning_optimizer.zero_grad()

    t2s_encoder_optimizer.zero_grad()
    t2s_decoder_optimizer.zero_grad()
    # print(s2t_loss.item(), dual_loss.item(), loss_num_pred.item())
    # print(loss_num_loc_pred.item(), loss_constant_pred.item(), s2t_semantic_loss.item())
    # print(t2s_loss.item(), t2s_semantic_loss.item())

    s2t_final_loss = s2t_loss + r1 * dual_loss + r3 * loss_num_pred + r4 * loss_num_loc_pred + r5 * loss_constant_pred + s2t_semantic_loss # 0.000001

    s2t_final_loss.backward(retain_graph=True)

    t2s_final_loss = t2s_loss + r6 * dual_loss + t2s_semantic_loss
    # print('t2s final loss:', t2s_final_loss.item())
    t2s_final_loss.backward()

    s2t_encoder_optimizer.step()
    s2t_predict_optimizer.step()
    s2t_generate_optimizer.step()
    s2t_merge_optimizer.step()
    ssl_num_loc_pred_optimizer.step()
    ssl_num_pred_optimizer.step()
    constant_reasoning_optimizer.step()

    t2s_encoder_optimizer.step()
    t2s_decoder_optimizer.step()

    # print("s2t loss: ", s2t_loss.item(), " t2s loss: ", t2s_loss.item(), " dual loss: ", dual_loss.item(),
    #       " s2t semantic loss: ", s2t_semantic_loss.item(), " t2s semantic loss: ", t2s_semantic_loss.item(),
    #       " num pred loss: ", loss_num_pred.item(), " num loc pred loss: ", loss_num_loc_pred.item(),
    #       " constant predict loss: ", loss_constant_pred.item(),
    #       " target_lm_prob_batch: ", np.mean(target_lm_prob_batch), " input_lm_prob_batch: ", np.mean(input_lm_prob_batch))


    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 0.5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 0.5)
    # torch.nn.utils.clip_grad_norm_(merge.parameters(), 0.5)

    return s2t_loss.item(),  t2s_loss.item(), dual_loss.item(), s2t_final_loss.item(), t2s_final_loss.item(), \
           loss_num_pred.item(), loss_num_loc_pred.item(), loss_constant_pred.item(), s2t_semantic_loss.item(), t2s_semantic_loss.item()


def evaluate_ns_solver(input_batch, input_length, generate_nums, encoder, predict, generate,
                                         merge, num_predict, num_loc_predict,constant_reasoning, output_lang, num_pos,
                                         beam_size=5, var_nums=[], beam_search=True, max_length=MAX_OUTPUT_LENGTH):
    # sequence mask for attention
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)


    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    num_predict.eval()
    num_loc_predict.eval()
    constant_reasoning.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()


    # Run words through encoder
    sigmoid = nn.Sigmoid()
    encoder_outputs, problem_output = encoder(input_var, [input_length])
    num_predict_logits = num_predict(encoder_outputs.transpose(0, 1))
    num_loc_predict_logits = num_loc_predict(encoder_outputs.transpose(0, 1))
    constant_prediction_logits = constant_reasoning(encoder_outputs.transpose(0, 1))
    # print(problem_output.size(), num_predict_logits.size(), num_loc_predict_logits.size(), constant_prediction_logits.size())

    rounded_constant_preds = torch.round(sigmoid(constant_prediction_logits))
    rounded_constant_preds = rounded_constant_preds.cpu().detach().numpy()

    # num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)+ len(var_nums)).fill_(0)
    num_mask = [0] * (len(num_pos) + len(generate_nums)+ len(var_nums))
    num = rounded_constant_preds.sum() + 3
    num_min = min(num, len(generate_nums))
    topv, topi = constant_prediction_logits.topk(int(num_min))
    for j in range(len(generate_nums)):
        if j not in topi:
            num_mask[j + len(var_nums)] = 1
    num_mask = torch.ByteTensor(num_mask)

    if USE_CUDA:
        num_mask = num_mask.cuda()

    # Prepare input and output variables  # # root embedding B x 1
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    # 提取与问题相关的数字embedding
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start - len(var_nums)
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    if beam_search:
        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                    b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                    seq_mask, num_mask)

                # leaf = p_leaf[:, 0].unsqueeze(1)
                # repeat_dims = [1] * leaf.dim()
                # repeat_dims[1] = op.size(1)
                # leaf = leaf.repeat(*repeat_dims)
                #
                # non_leaf = p_leaf[:, 1].unsqueeze(1)
                # repeat_dims = [1] * non_leaf.dim()
                # repeat_dims[1] = num_score.size(1)
                # non_leaf = non_leaf.repeat(*repeat_dims)
                #
                # p_leaf = torch.cat((leaf, non_leaf), dim=1)
                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                # out_score = p_leaf * out_score

                topv, topi = out_score.topk(beam_size)

                # is_leaf = int(topi[0])
                # if is_leaf:
                #     topv, topi = op.topk(1)
                #     out_token = int(topi[0])
                # else:
                #     topv, topi = num_score.topk(1)
                #     out_token = int(topi[0]) + num_start

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    # var_num当时数字处理，SEP/;当操作符处理
                    if out_token < num_start: # 非数字
                        generate_input = torch.LongTensor([out_token])
                        if USE_CUDA:
                            generate_input = generate_input.cuda()
                        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:  # 数字
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                                  current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break

        return beams[0].out
    else:
        all_node_outputs = []
        for t in range(max_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            out_scores = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
            out_tokens = torch.argmax(out_scores, dim=1) # B
            all_node_outputs.append(out_tokens)
            left_childs = []
            flag = False
            for idx, node_stack, out_token, embeddings_stack in zip(range(batch_size), node_stacks, out_tokens, embeddings_stacks):
                # node = node_stack.pop()
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    flag = True
                    break
                    # left_childs.append(None)
                    # continue
                # var_num当时数字处理，SEP/;当操作符处理
                if out_token < num_start: # 非数字
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
                    node_stack.append(TreeNode(right_child))
                    node_stack.append(TreeNode(left_child, left_flag=True))
                    embeddings_stack.append(TreeEmbedding(node_label.unsqueeze(0), False))
                else: # 数字
                    current_num = current_nums_embeddings[idx, out_token - num_start].unsqueeze(0)
                    while len(embeddings_stack) > 0 and embeddings_stack[-1].terminal:
                        sub_stree = embeddings_stack.pop()
                        op = embeddings_stack.pop()
                        current_num = merge(op.embedding.squeeze(0), sub_stree.embedding, current_num)
                    embeddings_stack.append(TreeEmbedding(current_num, terminal=True))

                if len(embeddings_stack) > 0 and embeddings_stack[-1].terminal:
                    left_childs.append(embeddings_stack[-1].embedding)
                else:
                    left_childs.append(None)
            if flag:
                break

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
        all_node_outputs = all_node_outputs.cpu().numpy()
        return all_node_outputs[0], sigmoid(num_predict_logits), sigmoid(num_loc_predict_logits), sigmoid(constant_prediction_logits)
