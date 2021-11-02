import sys
sys.path.append("../")
import math
import time
import torch.optim
from load_data import *
from num_transfer import *
from expression_tree import *
from log_utils import *
from prepare_data import *
from language_models import language_model
from language_models.train_and_evaluate import train_language_model, evaluate_language_model
from language_models.blue_metrics import Bleu
from language_models.rouge_metrics import Rouge


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)

# language_model_parameters = {
#     'model': 'LSTM', # type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)
#     'n_head': None,   # number of heads in the enc/dec of the Transformers
#     'emb_size': 32,     # size of the word embeddings
#     'n_hid': 64,     # number of hidden units per layer
#     'n_layers': 1,      # number of layers
#     'lr': 0.25,    # initial learning rate
#     'clip': 0.25,   # gradient clipping
#     'dropout_p': 0.05,    # dropout applied to layers
#     'tied': False,  # whether to tie the word embeddings and softmax weights
#     'log_interval': 100,
#     'epochs': 500,  # upper epoch limit
#     'batch_size': 128,
#     'seed': 1, # for reproducibility
#     'bptt': 2
# }

USE_CUDA = torch.cuda.is_available()
batch_size = 32
language_model_type = 'LSTM'
train_for_question = True #False #True
use_pos = True
n_heads = None
n_layers = 1
embedding_size = 128
hidden_size = 512
n_epochs = 500
learning_rate = 0.25
clip = 0.25
tied_weight = False
weight_decay = 0
dropout = 0.5
var_nums = []
ckpt_dir = "cm17k_lm" + '_' + str(batch_size) + '_' + str(hidden_size)
var_nums = ['x','y']
data_path = "../dataset/cm17k/questions_all.json"
prefix = '17k.json'
ori_path = '../dataset/cm17k/'


def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

def get_train_test_fold(ori_path,prefix,data,pairs):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item, pair in zip(data, pairs):
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold

save_dir = os.path.join("../models", ckpt_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if train_for_question:
    log_file = os.path.join(save_dir, 'lm_question_log')
else:
    log_file = os.path.join(save_dir, 'lm_equation_log')
create_logs(log_file)

data = load_cm17k_data(data_path)
pairs, generate_nums, copy_nums = transfer_cm17k_num(data)

temp_pairs = []
for p in pairs:
    # temp_pairs.append((p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]))
    temp_pairs.append(p)
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(ori_path, prefix, data, pairs)
print(len(train_fold), len(test_fold), len(valid_fold))

pairs_tested = test_fold
#pairs_trained = valid_fold
pairs_trained = train_fold

best_acc_fold = []
best_val_acc_fold = []
all_acc_data = []

input_lang, output_lang, pos_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                    copy_nums, tree=False)

# Initialize models
if train_for_question:
    if use_pos:
        if language_model_type == 'Transformer':
            lang_model = language_model.TransformerModel(vocab_size=pos_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                                                         n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        else:
            lang_model = language_model.RNNModel(vocab_size=pos_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                                                 n_layers=n_layers, embedding_dropout=dropout, rnn_dropout=dropout,
                                                 tie_weights=tied_weight, rnn_type=language_model_type)
    else:
        if language_model_type == 'Transformer':
            lang_model = language_model.TransformerModel(vocab_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                                                         n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        else:
            lang_model = language_model.RNNModel(vocab_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                                                 n_layers=n_layers, embedding_dropout=dropout, rnn_dropout=dropout,
                                                 tie_weights=tied_weight, rnn_type=language_model_type)
else:
    if language_model_type == 'Transformer':
        lang_model = language_model.TransformerModel(vocab_size=output_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                                                     n_layers=n_layers, n_heads=n_heads, dropout=dropout)
    else:
        lang_model = language_model.RNNModel(vocab_size=output_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                                             n_layers=n_layers, embedding_dropout=dropout, rnn_dropout=dropout,
                                             tie_weights=tied_weight, rnn_type=language_model_type)

# optimizer
lang_model_optimizer = torch.optim.Adam(lang_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# opt scheduler
lang_model_scheduler = torch.optim.lr_scheduler.StepLR(lang_model_optimizer, step_size=20, gamma=0.5)

# Move models to GPU
if USE_CUDA:
    lang_model.cuda()

current_save_dir = save_dir
rouge = Rouge()
bleu_1 = Bleu(n=1)
bleu_2 = Bleu(n=2)
bleu_3 = Bleu(n=3)
bleu_4 = Bleu(n=4)
best_ppl = sys.maxsize
for epoch in range(n_epochs):
    loss_total = 0
    # input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
    # num_pos_batches, num_size_batches, ans_batches, id_batches, type_batches, pos_batches, pos_lengths = prepare_train_batch(train_pairs, batch_size)

    batches_dict = prepare_train_batch(train_pairs, batch_size)

    if train_for_question:
        if use_pos:
            batches_dict["train_batches"] = batches_dict["pos_batches"]
            batches_dict["train_lengths"]= batches_dict["pos_lengths"]
            lang = pos_lang
        else:
            batches_dict["train_batches"] = batches_dict["input_batches"]
            batches_dict["train_lengths"] = batches_dict["input_lengths"]
            lang = input_lang
    else:
        batches_dict["train_batches"] = batches_dict["output_batches"]
        batches_dict["train_lengths"] = batches_dict["output_lengths"]
        lang = output_lang

    logs_content = "epoch: {}".format(epoch + 1)
    add_log(log_file, logs_content)
    start = time.time()
    input_lengths = batches_dict["input_lengths"]
    for idx in range(len(input_lengths)):
        loss = train_language_model(batches_dict["train_batches"][idx], batches_dict["train_lengths"][idx], lang_model,
                                    lang_model_optimizer, clip=clip, model_type=language_model_type)

        loss_total += loss
    lang_model_scheduler.step()
    train_loss = loss_total / len(input_lengths)
    train_ppl = np.exp(loss_total / len(input_lengths))
    logs_content = "loss: {} ppl: {}".format(train_loss, train_ppl)
    add_log(log_file, logs_content)
    logs_content = "training time: {}".format(time_since(time.time() - start))
    add_log(log_file, logs_content)
    logs_content = "--------------------------------"
    add_log(log_file, logs_content)
    if epoch % 1 == 0 or epoch > n_epochs - 5:
        total_loss = 0
        start = time.time()
        # input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
        # num_pos_batches, num_size_batches, ans_batches, id_batches, type_batches, pos_batches, pos_lengths = prepare_train_batch(test_pairs, 1)
        batches_dict = prepare_train_batch(test_pairs, 1)
        if train_for_question:
            if use_pos:
                batches_dict["test_batches"] = batches_dict["pos_batches"]
                batches_dict["test_lengths"] = batches_dict["pos_lengths"]
                lang = pos_lang
            else:
                batches_dict["test_batches"] = batches_dict["input_batches"]
                batches_dict["test_lengths"] = batches_dict["input_lengths"]
                lang = input_lang
        else:
            batches_dict["test_batches"] = batches_dict["output_batches"]
            batches_dict["test_lengths"] = batches_dict["output_lengths"]
            lang = output_lang

        ref_list = []
        gen_list = []
        input_lengths = batches_dict["input_lengths"]
        for idx in range(len(input_lengths)):
            loss, test_output = evaluate_language_model(batches_dict["test_batches"][idx],  batches_dict["test_lengths"][idx], lang_model, model_type=language_model_type)
            total_loss += loss
            for test, ref in zip(test_output, batches_dict["test_batches"][idx]):
                test_list = []
                for t in test:
                    test_list.append(lang.index2word[t])
                gen_list.append(' '.join(test_list))

                tar_list = []
                for r in ref:
                    tar_list.append(lang.index2word[r])
                ref_list.append(' '.join(tar_list))
        average_rouge_score, rouge_scores = rouge.compute_score(gen_list, ref_list)
        average_bleu_1_score, bleu_1_scores = bleu_1.compute_score(gen_list, ref_list)
        average_bleu_2_score, bleu_2_scores = bleu_1.compute_score(gen_list, ref_list)
        average_bleu_3_score, bleu_3_scores = bleu_1.compute_score(gen_list, ref_list)
        average_bleu_4_score, bleu_4_scores = bleu_1.compute_score(gen_list, ref_list)

        test_loss = total_loss / len(input_lengths)
        test_ppl = np.exp(total_loss / len(input_lengths))
        logs_content = "loss: {} ppl: {}".format(test_loss, test_ppl)
        add_log(log_file, logs_content)
        logs_content = "Rogue-L: {}".format(average_rouge_score)
        add_log(log_file, logs_content)
        logs_content = "BLUE-1: {}  BLUE-2: {} BLUE-2: {} BLUE-3: {}".format(average_bleu_1_score[0], average_bleu_2_score[0], average_bleu_3_score[0], average_bleu_3_score[0])
        add_log(log_file, logs_content)
        logs_content = "valid time: {}".format(time_since(time.time() - start))
        add_log(log_file, logs_content)
        logs_content = "--------------------------------"
        add_log(log_file, logs_content)

        name_for_save = "language_model_for_question" if train_for_question else "language_model_for_equation_template"
        # torch.save(lang_model.state_dict(), os.path.join(current_save_dir, name_for_save))
        torch.save(lang_model, os.path.join(current_save_dir, name_for_save))
        if test_ppl < best_ppl:
            # torch.save(lang_model.state_dict(), os.path.join(current_save_dir, name_for_save + '_ppl_best'))
            torch.save(lang_model, os.path.join(current_save_dir, name_for_save + '_ppl_best'))
            best_ppl = test_ppl
logs_content = "Best PPL: {}".format(best_ppl)
add_log(log_file, logs_content)
