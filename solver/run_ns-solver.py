import sys
sys.path.append('../')
import os
from train_and_evaluate import *
# from models import *
from models import *
from prepare_data import *
import time
import torch.optim
from load_data import *
from num_transfer import *
from expression_tree import *
from log_utils import *
from language_models.inference_language_model import LMProb
from calculate import *
from metrics import binary_accuracy, exact_match, inclusion_match, inclusion_match_with_addon

USE_CUDA = torch.cuda.is_available()
batch_size = 32
embedding_size = 128
hidden_size = 512
n_epochs = 160
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
dropout_rate = 0.5
beam_search = False
fold_num = 1
random_seed = 1
use_pos = True
dataset_name = "cm17k"
r1 = 0.0005
r2 = 0.01
r3 = 1.0
r4 = 1.0
r5 = 1.0
r6 = 0.005
r7 = 0.1
ckpt_dir = "cm17k" + '_' + str(batch_size) + '_' + str(hidden_size)
var_nums = ['x','y']
data_path = "../dataset/cm17k/questions_all.json"
prefix = '17k.json'
ori_path = '../dataset/cm17k/'


def read_json(path):
    with open(path,'r', encoding='utf-8') as f:
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

log_file = os.path.join(save_dir, 'log')
create_logs(log_file)

data = load_cm17k_data(data_path)
pairs, generate_nums, copy_nums, max_problem_len = transfer_cm17k_num_all(data)

# pairs: (id, type, input_seq, pos_seq, out_seq, nums, num_pos, sni_nums, sni_num_pos, unks, constants, ops, ans)
temp_pairs = []
for p in pairs:
    ept = ExpressionTree()
    ept.build_tree_from_infix_expression(p["out_seq"])
    p['out_seq'] = ept.get_prefix_expression()
    temp_pairs.append(p)
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(ori_path, prefix, data, pairs)
print(len(train_fold), len(test_fold), len(valid_fold))

pairs_tested = test_fold
# pairs_valid = valid_fold
pairs_trained = train_fold

last_acc_fold_for_val = []
best_val_acc_fold_for_val = []
all_acc_data_for_val = []
last_acc_fold_for_test = []
best_val_acc_fold_for_test = []
all_acc_data_for_test = []


current_save_dir = save_dir
lang_model_for_question = LMProb(current_save_dir + os.path.sep + "language_model_for_question_ppl_best")
lang_model_for_equation = LMProb(current_save_dir + os.path.sep + "language_model_for_equation_template_ppl_best")

# id type input_seq, input_len, pos_seq, pos_len, output_seq, output_len, nums, num_pos, num_stack,
# sni_nums, sni_num_pos, unks, constants, ops, ans
input_lang, output_lang, pos_lang, train_pairs, test_pairs = prepare_data_all(pairs_trained, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True)
# print(input_lang.word2index)
new_train_pairs = []
new_test_pairs = []
for pair in train_pairs:
    if use_pos:
        pair['input_lm_prob'] = lang_model_for_question.get_prob(pair['pos_cell'])
    else:
        pair['input_lm_prob'] = lang_model_for_question.get_prob(pair['input_cell'])
    pair['eq_lm_prob'] = lang_model_for_equation.get_prob(pair['output_cell'])
    new_train_pairs.append(pair)
    # print(pair['input_lm_prob'], pair['eq_lm_prob'])
    # if torch.isinf(pair['input_lm_prob']) or torch.isinf(pair['eq_lm_prob']):
    #     exit(1)

for pair in test_pairs:
    if use_pos:
        pair['input_lm_prob'] = lang_model_for_question.get_prob(pair['pos_cell'])
    else:
        pair['input_lm_prob'] = lang_model_for_question.get_prob(pair['input_cell'])
    pair['eq_lm_prob'] = lang_model_for_equation.get_prob(pair['output_cell'])
    new_test_pairs.append(pair)

train_pairs = new_train_pairs
test_pairs = new_test_pairs

# Initialize models
s2t_encoder = Seq2TreeEncoder(vocab_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                         n_layers=n_layers, embedding_dropout=dropout_rate, rnn_dropout=dropout_rate)

num_predict_loss_fn = nn.BCEWithLogitsLoss()
num_loc_predict_loss_fn = nn.BCEWithLogitsLoss()
constant_reasoning_loss_fn = nn.BCEWithLogitsLoss()
num_predict = SelfSupervisionNumberPrediction(classes_size=copy_nums, hidden_size=hidden_size, dropout=dropout_rate)
num_loc_predict = SelfSupervisionNumberLocationPrediction(classes_size=max_problem_len, hidden_size=hidden_size, dropout=dropout_rate)
constant_reasoning = ConstantReasoning(classes_size=len(generate_nums), hidden_size=hidden_size, dropout=dropout_rate)


s2t_predict = Seq2TreePrediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(var_nums),
                             vocab_size=len(generate_nums) + len(var_nums), dropout=dropout_rate)
s2t_generate = Seq2TreeNodeGeneration(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size, dropout=dropout_rate)
s2t_merge = Seq2TreeSubTreeMerge(hidden_size=hidden_size, embedding_size=embedding_size, dropout=dropout_rate)

# t2s_encoder = None
# t2s_decoder = None
t2s_encoder = ExplicitTreeEncoder(output_lang, embedding_size, hidden_size, dropout=dropout_rate)
if use_pos:
    classes_size = pos_lang.n_words
    vocab_size = pos_lang.n_words
else:
    classes_size = input_lang.n_words
    vocab_size = input_lang.n_words
t2s_decoder = GeneralAttnDecoderRNN(vocab_size=vocab_size, classes_size=classes_size,
                  embedding_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers,
                                    rnn_dropout=dropout_rate, embedding_dropout=dropout_rate)

# the embedding layer is  only for generated number embeddings, operators, and paddings

s2t_encoder_optimizer = torch.optim.Adam(s2t_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
s2t_predict_optimizer = torch.optim.Adam(s2t_predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
s2t_generate_optimizer = torch.optim.Adam(s2t_generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
s2t_merge_optimizer = torch.optim.Adam(s2t_merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
t2s_encoder_optimizer = torch.optim.Adam(t2s_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
t2s_decoder_optimizer = torch.optim.Adam(t2s_decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
num_predict_optimizer = torch.optim.Adam(num_predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
num_loc_predict_optimizer = torch.optim.Adam(num_loc_predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
constant_reasoning_optimizer = torch.optim.Adam(constant_reasoning.parameters(), lr=learning_rate, weight_decay=weight_decay)

s2t_encoder_scheduler = torch.optim.lr_scheduler.StepLR(s2t_encoder_optimizer, step_size=max(1, n_epochs//4), gamma=0.5)
s2t_predict_scheduler = torch.optim.lr_scheduler.StepLR(s2t_predict_optimizer, step_size=max(1, n_epochs//4), gamma=0.5)
s2t_generate_scheduler = torch.optim.lr_scheduler.StepLR(s2t_generate_optimizer, step_size=max(1, n_epochs//4), gamma=0.5)
s2t_merge_scheduler = torch.optim.lr_scheduler.StepLR(s2t_merge_optimizer, step_size=max(1, n_epochs//4), gamma=0.5)
t2s_encoder_scheduler = torch.optim.lr_scheduler.StepLR(t2s_encoder_optimizer, step_size=max(1, n_epochs//4), gamma=0.5)
t2s_decoder_scheduler = torch.optim.lr_scheduler.StepLR(t2s_decoder_optimizer, step_size=max(1, n_epochs//4), gamma=0.5)
num_predict_scheduler = torch.optim.lr_scheduler.StepLR(num_predict_optimizer, step_size=max(1, n_epochs//4), gamma=0.5)
num_loc_predict_scheduler = torch.optim.lr_scheduler.StepLR(num_loc_predict_optimizer, step_size=max(1, n_epochs//4), gamma=0.5)
constant_reasoning_scheduler = torch.optim.lr_scheduler.StepLR(constant_reasoning_optimizer, step_size=max(1, n_epochs//4), gamma=0.5)


# Move models to GPU
if USE_CUDA:
    s2t_encoder.cuda()
    s2t_predict.cuda()
    s2t_generate.cuda()
    s2t_merge.cuda()
    t2s_encoder.cuda()
    t2s_decoder.cuda()
    num_predict = num_predict.cuda()
    num_loc_predict = num_loc_predict.cuda()
    constant_reasoning = constant_reasoning.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

var_num_ids = []
for var in var_nums:
    if var in output_lang.word2index.keys():
        var_num_ids.append(output_lang.word2index[var])


best_val_acc_for_test = 0
best_equ_acc_for_test = 0
current_best_val_acc_for_test = (0, 0, 0)

for epoch in range(n_epochs):
    # loss_total = 0
    start = time.time()
    s2t_loss_total,  t2s_loss_total, dual_loss_total, s2t_final_loss_total, \
    t2s_final_loss_total, loss_num_pred_total, loss_num_loc_pred_total, \
    loss_constant_pred_total, s2t_semantic_loss_total, t2s_semantic_loss_total = 0,0,0,0,0,0,0,0,0,0

    batches_dict = prepare_train_batch_all_for_dual(train_pairs, batch_size)
    id_batches = batches_dict['id_batches']
    type_batches = batches_dict['type_batches']
    input_batches = batches_dict['input_batches']
    input_length_batches = batches_dict['input_length_batches']
    pos_batches = batches_dict['pos_batches']
    pos_length_batches = batches_dict['pos_length_batches']
    output_batches = batches_dict['output_batches']
    output_length_batches = batches_dict['output_length_batches']
    nums_batches = batches_dict['nums_batches']
    num_pos_batches = batches_dict['num_pos_batches']
    num_size_batches = batches_dict['num_size_batches']
    num_stack_batches = batches_dict['num_stack_batches']
    sni_nums_batches = batches_dict['sni_nums_batches']
    sni_num_pos_batches = batches_dict['sni_num_pos_batches']
    sni_num_size_batches = batches_dict['sni_num_size_batches']
    vars_batches = batches_dict['vars_batches']
    var_size_batches = batches_dict['var_size_batches']
    constants_batches = batches_dict['constants_batches']
    constant_size_batches = batches_dict['constant_size_batches']
    opses_batches = batches_dict['opses_batches']
    input_lm_prob_batches = batches_dict['input_lm_prob_batches']
    eq_lm_prob_batches = batches_dict['eq_lm_prob_batches']
    ans_batches = batches_dict['ans_batches']

    logs_content = "epoch: {}".format(epoch + 1)
    add_log(log_file,logs_content)
    for idx in range(len(input_length_batches)):
        if use_pos:
            dual_target_batch = pos_batches[idx]
            dual_target_length = pos_length_batches[idx]
            dual_output_lang = pos_lang
        else:
            dual_target_batch = input_batches[idx]
            dual_target_length = pos_length_batches[idx]
            dual_output_lang = input_lang
        s2t_loss,  t2s_loss, dual_loss, s2t_final_loss, t2s_final_loss, loss_num_pred, loss_num_loc_pred, \
        loss_constant_pred, s2t_semantic_loss, t2s_semantic_loss = train_ns_solver(input_batches[idx], input_length_batches[idx], output_batches[idx],
                           output_length_batches[idx], dual_target_batch, dual_target_length,
                           num_stack_batches[idx], num_size_batches[idx], generate_num_ids,
                           input_lm_prob_batches[idx], eq_lm_prob_batches[idx],
                           s2t_encoder, s2t_predict, s2t_generate, s2t_merge, t2s_encoder,
                           t2s_decoder, num_predict, num_loc_predict,constant_reasoning,
                           s2t_encoder_optimizer, s2t_predict_optimizer,
                           s2t_generate_optimizer, s2t_merge_optimizer, t2s_encoder_optimizer, t2s_decoder_optimizer,
                           num_predict_optimizer, num_loc_predict_optimizer, constant_reasoning_optimizer,
                           output_lang, dual_output_lang, num_predict_loss_fn,
                           num_loc_predict_loss_fn, constant_reasoning_loss_fn, max_problem_len, copy_nums,
                           constants_batches[idx], num_pos_batches[idx], var_nums=var_num_ids, use_teacher_forcing=0.83,
                           r1=r1, r2=r2, r3=r3, r4=r4, r5=r5, r6=r6, r7=r7)

        s2t_loss_total += s2t_loss
        t2s_loss_total += t2s_loss
        dual_loss_total += dual_loss
        s2t_final_loss_total += s2t_final_loss
        t2s_final_loss_total += t2s_final_loss
        loss_num_pred_total += loss_num_pred
        loss_num_loc_pred_total += loss_num_loc_pred
        loss_constant_pred_total += loss_constant_pred
        s2t_semantic_loss_total += s2t_semantic_loss
        t2s_semantic_loss_total += t2s_semantic_loss

    s2t_encoder_scheduler.step()
    s2t_predict_scheduler.step()
    s2t_generate_scheduler.step()
    s2t_merge_scheduler.step()
    num_predict_scheduler.step()
    num_loc_predict_scheduler.step()
    constant_reasoning_scheduler.step()

    t2s_encoder_scheduler.step()
    t2s_decoder_scheduler.step()


    logs_content = "s2t loss: {} t2s loss: {} dual loss: {} s2t semantic loss: {} t2s semantic loss: {}" \
                   "s2t final loss: {} t2s final loss: {} loss_num_pred: {} loss_num_loc_pred: {} loss_constant_pred: {}".format(s2t_loss_total / len(input_length_batches),
                                                                  t2s_loss_total / len(input_length_batches),
                                                                  dual_loss_total / len(input_length_batches),
                                                                  s2t_semantic_loss_total / len(input_length_batches),
                                                                  t2s_semantic_loss_total / len(input_length_batches),
                                                                  s2t_final_loss_total / len(input_length_batches),
                                                                  t2s_final_loss_total / len(input_length_batches),
                                                                  loss_num_pred_total / len(input_length_batches),
                                                                  loss_num_loc_pred_total / len(input_length_batches),
                                                                  loss_constant_pred_total / len(input_length_batches))

    add_log(log_file,logs_content)
    logs_content = "training time: {}".format(time_since(time.time() - start))
    add_log(log_file,logs_content)
    logs_content = "--------------------------------"
    if epoch % 1 == 0 or epoch > n_epochs - 5:
        value_ac_for_test = 0
        equation_ac_for_test = 0
        answer_ac_for_test = 0
        eval_total_for_test = 0
        constant_binary_acc_for_test = 0
        constant_exact_acc_for_test = 0
        constant_inclusion_acc_for_test = 0
        constant_inclusion_addon_acc_for_test = 0
        num_binary_acc_for_test = 0
        num_loc_binary_acc_for_test = 0
        num_loc_exact_acc_for_test = 0
        num_loc_inclusion_acc_for_test = 0
        num_loc_inclusion_addon_acc_for_test = 0
        start = time.time()

        for test_batch in test_pairs:
            # id, type, input_seq, input_len, pos_seq, pos_len, output_seq, output_len, nums, num_pos, num_stack, sni_nums, sni_num_pos, unks, constants, ops, input_lm_prob, eq_lm_prob, ans))
            test_res, num_prediction_probabilty, num_loc_prediction_probabilty, constant_prediction_probabilty = \
                evaluate_ns_solver(test_batch['input_cell'], test_batch['input_cell_len'], generate_num_ids, s2t_encoder,
                                   s2t_predict, s2t_generate, s2t_merge, num_predict,
                                   num_loc_predict, constant_reasoning, output_lang,
                                   test_batch['num_pos'], beam_size=beam_size, beam_search=beam_search,
                                   var_nums=var_num_ids)
            try:
                val_ac, equ_ac, ans_ac, _, _ = compute_equations_result(test_res, test_batch["output_cell"], output_lang,
                                                                        test_batch["nums"], test_batch["num_stack"],
                                                                        ans_list=test_batch["ans"], tree=True, prefix=True)
            except Exception as e:
                # print(e)
                val_ac, equ_ac, ans_ac = False, False, False
            if val_ac:
                value_ac_for_test += 1
            if ans_ac:
                answer_ac_for_test += 1
            if equ_ac:
                equation_ac_for_test += 1
            eval_total_for_test += 1
            # constant
            constants = []
            for constant in test_batch["constants"]:
                constants.append(constant-output_lang.num_start)
            constant_binary_acc_for_test += float(binary_accuracy(constant_prediction_probabilty, constants, max_len=len(generate_nums)).cpu().numpy())
            constant_exact_acc_for_test += exact_match(constant_prediction_probabilty, constants, max_len=len(generate_nums))
            constant_inclusion_acc_for_test += inclusion_match(constant_prediction_probabilty, constants, max_len=len(generate_nums))
            constant_inclusion_addon_acc_for_test += inclusion_match_with_addon(constant_prediction_probabilty, constants, max_len=len(generate_nums), add_on=3)
            # num and num pos
            num_binary_acc_for_test += float(binary_accuracy(num_prediction_probabilty, [len(test_batch["num_pos"])], max_len=copy_nums).cpu().numpy())

            num_loc_binary_acc_for_test += float(binary_accuracy(num_loc_prediction_probabilty, test_batch["num_pos"], max_len=max_problem_len).cpu().numpy())
            num_loc_exact_acc_for_test += exact_match(num_loc_prediction_probabilty, test_batch["num_pos"], max_len=max_problem_len)
            num_loc_inclusion_acc_for_test += inclusion_match(num_loc_prediction_probabilty, test_batch["num_pos"], max_len=max_problem_len)
            num_loc_inclusion_addon_acc_for_test += inclusion_match_with_addon(num_loc_prediction_probabilty, test_batch["num_pos"], max_len=max_problem_len, add_on=3)


        logs_content = "Test: Num: binary acc: {}".format(num_binary_acc_for_test / eval_total_for_test,)
        add_log(log_file, logs_content)
        logs_content = "Test: Num Loc: binary acc: {} exact acc: {} inclusion acc: {} inclusion addon acc: {}".format(num_loc_binary_acc_for_test / eval_total_for_test,
                                                                                                                       num_loc_exact_acc_for_test / eval_total_for_test,
                                                                                                                       num_loc_inclusion_acc_for_test / eval_total_for_test,
                                                                                                                       num_loc_inclusion_addon_acc_for_test / eval_total_for_test)
        add_log(log_file, logs_content)
        logs_content = "Test: Constant: binary acc: {} exact acc: {} inclusion acc: {} inclusion addon acc: {}".format(constant_binary_acc_for_test / eval_total_for_test,
                                                                                                                        constant_exact_acc_for_test / eval_total_for_test,
                                                                                                                        constant_inclusion_acc_for_test / eval_total_for_test,
                                                                                                                        constant_inclusion_addon_acc_for_test / eval_total_for_test)
        add_log(log_file, logs_content)
        logs_content = "Test: {}, {}, {}".format(equation_ac_for_test, value_ac_for_test, eval_total_for_test)
        add_log(log_file, logs_content)
        logs_content = "test_answer_acc: {} {}".format(float(equation_ac_for_test) / eval_total_for_test, float(value_ac_for_test) / eval_total_for_test)
        add_log(log_file, logs_content)

        logs_content = "testing time: {}".format(time_since(time.time() - start))
        add_log(log_file, logs_content)
        logs_content = "------------------------------------------------------"
        add_log(log_file, logs_content)
        all_acc_data_for_test.append((epoch, equation_ac_for_test, value_ac_for_test, eval_total_for_test))
        # torch.save(encoder.state_dict(), "models/seq2tree_encoder")
        # torch.save(predict.state_dict(), "models/seq2tree_predict")
        # torch.save(generate.state_dict(), "models/seq2tree_generate")
        # torch.save(merge.state_dict(), "models/seq2tree_merge")
        torch.save(s2t_encoder.state_dict(), os.path.join(current_save_dir, "seq2tree_encoder"))
        torch.save(s2t_predict.state_dict(), os.path.join(current_save_dir, "seq2tree_predict"))
        torch.save(s2t_generate.state_dict(), os.path.join(current_save_dir, "seq2tree_generate"))
        torch.save(s2t_merge.state_dict(),  os.path.join(current_save_dir, "seq2tree_merge"))
        torch.save(t2s_encoder.state_dict(), os.path.join(current_save_dir, "tree2seq_encoder"))
        torch.save(t2s_decoder.state_dict(), os.path.join(current_save_dir, "tree2seq_decoder"))
        torch.save(num_predict.state_dict(),  os.path.join(current_save_dir, "seq2tree_num_predict"))
        torch.save(num_loc_predict.state_dict(),  os.path.join(current_save_dir, "seq2tree_num_loc_predict"))
        torch.save(constant_reasoning.state_dict(),  os.path.join(current_save_dir, "seq2tree_constant_reasoning"))

        if best_val_acc_for_test < value_ac_for_test:
            best_val_acc_for_test = value_ac_for_test
            current_best_val_acc_for_test = (equation_ac_for_test, value_ac_for_test, eval_total_for_test)
            torch.save(s2t_encoder.state_dict(), os.path.join(current_save_dir, "seq2tree_encoder_best_val_acc_for_test"))
            torch.save(s2t_predict.state_dict(), os.path.join(current_save_dir, "seq2tree_predict_best_val_acc_for_test"))
            torch.save(s2t_generate.state_dict(), os.path.join(current_save_dir, "seq2tree_generate_best_val_acc_for_test"))
            torch.save(s2t_merge.state_dict(),  os.path.join(current_save_dir, "seq2tree_merge_best_val_acc_for_test"))
            torch.save(t2s_encoder.state_dict(), os.path.join(current_save_dir, "tree2seq_encoder_best_val_acc_for_test"))
            torch.save(t2s_decoder.state_dict(), os.path.join(current_save_dir, "tree2seq_decoder_best_val_acc_for_test"))
            torch.save(num_predict.state_dict(),  os.path.join(current_save_dir, "seq2tree_num_predict_best_val_acc_for_test"))
            torch.save(num_loc_predict.state_dict(),  os.path.join(current_save_dir, "seq2tree_num_loc_predict_best_val_acc_for_test"))
            torch.save(constant_reasoning.state_dict(),  os.path.join(current_save_dir, "seq2tree_constant_reasoning_best_val_acc_for_test"))
        if best_equ_acc_for_test < equation_ac_for_test:
            best_equ_acc_for_test = equation_ac_for_test
            torch.save(s2t_encoder.state_dict(), os.path.join(current_save_dir, "seq2tree_encoder_best_equ_acc_for_test"))
            torch.save(s2t_predict.state_dict(), os.path.join(current_save_dir, "seq2tree_predict_best_equ_acc_for_test"))
            torch.save(s2t_generate.state_dict(), os.path.join(current_save_dir, "seq2tree_generate_best_equ_acc_for_test"))
            torch.save(s2t_merge.state_dict(),  os.path.join(current_save_dir, "seq2tree_merge_best_equ_acc_for_test"))
            torch.save(t2s_encoder.state_dict(), os.path.join(current_save_dir, "tree2seq_encoder_best_equ_acc_for_test"))
            torch.save(t2s_decoder.state_dict(), os.path.join(current_save_dir, "tree2seq_decoder_best_equ_acc_for_test"))
            torch.save(num_predict.state_dict(),  os.path.join(current_save_dir, "seq2tree_num_predict_best_equ_acc_for_test"))
            torch.save(num_loc_predict.state_dict(),  os.path.join(current_save_dir, "seq2tree_num_loc_predict_best_equ_acc_for_test"))
            torch.save(constant_reasoning.state_dict(),  os.path.join(current_save_dir, "seq2tree_constant_reasoning_best_equ_acc_for_test"))
        if epoch == n_epochs - 1:
            last_acc_fold_for_test.append((equation_ac_for_test, value_ac_for_test, eval_total_for_test))
            best_val_acc_fold_for_test.append(current_best_val_acc_for_test)

a, b, c = 0, 0, 0
for bl in range(len(last_acc_fold_for_test)):
    a += last_acc_fold_for_test[bl][0]
    b += last_acc_fold_for_test[bl][1]
    c += last_acc_fold_for_test[bl][2]
    # print(best_acc_fold[bl])
    logs_content = "{}".format(last_acc_fold_for_test[bl])
    add_log(log_file, logs_content)
# print(a / float(c), b / float(c))
logs_content = "{} {}".format(a / float(c), b / float(c))
add_log(log_file, logs_content)
# print("------------------------------------------------------")
logs_content = "------------------------------------------------------"
add_log(log_file, logs_content)

a, b, c = 0, 0, 0
for bl in range(len(best_val_acc_fold_for_test)):
    a += best_val_acc_fold_for_test[bl][0]
    b += best_val_acc_fold_for_test[bl][1]
    c += best_val_acc_fold_for_test[bl][2]
    # print(best_val_acc_fold[bl])
    logs_content = "{}".format(best_val_acc_fold_for_test[bl])
    add_log(log_file, logs_content)
# print(a / float(c), b / float(c))
logs_content = "{} {}".format(a / float(c), b / float(c))
add_log(log_file, logs_content)
# print("------------------------------------------------------")
logs_content = "------------------------------------------------------"
add_log(log_file, logs_content)

logs_content = "{}".format(all_acc_data_for_test)
add_log(log_file, logs_content)


