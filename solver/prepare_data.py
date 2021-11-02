import random
import json
import copy
import re
import nltk
from data_utils import remove_brackets
from lang import InputLang, OutputLang, POSLang
from data_utils import indexes_from_sentence, pad_seq, check_bracket, get_num_stack


# pairs: (input_seq, eq_segs, nums, num_pos, ans, id, type, pos_seq)
def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = InputLang()
    output_lang = OutputLang()
    pos_lang = POSLang()
    train_pairs = []
    test_pairs = []

    print("Indexing words")
    for pair in pairs_trained:
        if len(pair["num_pos"]) > 0:
            input_lang.add_sen_to_vocab(pair["input_seq"])
            output_lang.add_sen_to_vocab(pair["out_seq"])
            pos_lang.add_sen_to_vocab(pair["pos_seq"])
    input_lang.build_input_lang(trim_min_count)
    pos_lang.build_lang(trim_min_count=1)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []  # 用于记录不在输出词典的数字
        for word in pair["out_seq"]:
            temp_num = []
            flag_not = True  # 用检查等式是否存在不在字典的元素
            if word not in output_lang.index2word:  # 如果该元素不在输出字典里
                flag_not = False
                for i, j in enumerate(pair["nums"]): # 遍历nums, 看是否存在
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair["nums"]))])  # 生成从0到等式长度的数字

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair["input_seq"])
        output_cell = indexes_from_sentence(output_lang, pair["out_seq"], tree)
        pos_cell = indexes_from_sentence(pos_lang, pair["pos_seq"])
        # pairs: (input_seq, eq_segs, nums, num_pos, ans, id, type, pos_seq)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], pair[4], num_stack, pair[5], pair[6], pos_cell, len(pos_cell)))
        train_dict = {
            "id": pair['id'],
            "type": pair['type'],
            "input_cell": input_cell,
            "input_cell_len": len(input_cell),
            "output_cell": output_cell,
            "output_cell_len": len(output_cell),
            "pos_cell": pos_cell,
            "pos_cell_len": len(pos_cell),
            "nums": pair['nums'],
            "num_pos": pair['num_pos'],
            "num_stack": num_stack,
            "ans": pair['ans']
        }
        train_pairs.append(train_dict)

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    for pair in pairs_tested:
        num_stack = []
        for word in pair["out_seq"]:  # out_seq
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word: # 非符号，即word为数字
                flag_not = False
                for i, j in enumerate(pair["nums"]): # nums
                    if j == word:
                        temp_num.append(i) # 在等式的位置信息

            if not flag_not and len(temp_num) != 0:# 数字在数字列表中
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                # 数字不在数字列表中，则生成数字列表长度的位置信息，
                # 生成时根据解码器的概率选一个， 参见generate_tree_input
                num_stack.append([_ for _ in range(len(pair["nums"]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair["input_seq"])
        output_cell = indexes_from_sentence(output_lang, pair["out_seq"], tree)
        pos_cell = indexes_from_sentence(pos_lang, pair["pos_seq"])
        # test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], pair[4], num_stack, pair[5], pair[6], pos_cell, len(pos_cell)))
        test_dict = {
            "id": pair['id'],
            "type": pair['type'],
            "input_cell": input_cell,
            "input_cell_len": len(input_cell),
            "output_cell": output_cell,
            "output_cell_len": len(output_cell),
            "pos_cell": pos_cell,
            "pos_cell_len": len(pos_cell),
            "nums": pair['nums'],
            "num_pos": pair['num_pos'],
            "num_stack": num_stack,
            "ans": pair['ans']
        }
        test_pairs.append(test_dict)
    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, pos_lang, train_pairs, test_pairs

# pairs: (id, type, input_seq, pos_seq, out_seq, nums, num_pos, sni_nums, sni_num_pos, unks, constants, ops, ans)
def prepare_data_all(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = InputLang()
    output_lang = OutputLang()
    pos_lang = POSLang()
    train_pairs = []
    test_pairs = []

    print("Indexing words")
    for pair in pairs_trained:
        if len(pair["nums"]) > 0:
            input_lang.add_sen_to_vocab(pair["input_seq"])
            pos_lang.add_sen_to_vocab(pair["pos_seq"])
            output_lang.add_sen_to_vocab(pair["out_seq"])

    input_lang.build_input_lang(trim_min_count)
    pos_lang.build_lang(trim_min_count=1)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []  # 用于记录不在输出词典的数字
        for word in pair["out_seq"]:
            temp_num = []
            flag_not = True  # 用检查等式是否存在不在字典的元素
            if word not in output_lang.index2word:  # 如果该元素不在输出字典里
                flag_not = False
                for i, j in enumerate(pair["nums"]): # 遍历nums, 看是否存在
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair["nums"]))])  # 生成从0到等式长度的数字

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair["input_seq"])
        pos_cell = indexes_from_sentence(pos_lang, pair["pos_seq"])
        output_cell = indexes_from_sentence(output_lang, pair["out_seq"], tree)

        constants = []
        for constant in pair["constants"]:
            if constant in output_lang.word2index.keys():
                constants.append(output_lang.word2index[constant])

        opses = []
        for ops in pair["ops"]:
            if ops in ['[',']','(',')'] and tree:
                continue
            else:
                if ops == ';':
                    opses.append(output_lang.word2index['SEP'])
                else:
                    opses.append(output_lang.word2index[ops])

        vars = []
        for var in pair["unks"]:
            if var in output_lang.word2index.keys():
                vars.append(output_lang.word2index[var])


        # id type input_seq, input_len, pos_seq, pos_len, output_seq, output_len, nums, num_pos, num_stack,
        # sni_nums, sni_num_pos, unks, constants, ops, ans

        # train_pairs.append((pair[0], pair[1], input_cell, len(input_cell), pos_cell, len(pos_cell), output_cell, len(output_cell),
        #                     pair[5], pair[6], num_stack, pair[7], pair[8], vars, constants, opses, pair[12],))

        # pairs: (id, type, input_seq, pos_seq, out_seq, nums, num_pos, sni_nums, sni_num_pos, unks, constants, ops, ans)
        train_dict = {
            "id": pair['id'],
            "type": pair['type'],
            "input_cell": input_cell,
            "input_cell_len": len(input_cell),
            "output_cell": output_cell,
            "output_cell_len": len(output_cell),
            "pos_cell": pos_cell,
            "pos_cell_len": len(pos_cell),
            "nums": pair['nums'],
            "num_pos": pair['num_pos'],
            "num_stack": num_stack,
            "sni_nums": pair['sni_nums'],
            "sni_num_pos": pair['sni_num_pos'],
            'vars': vars,
            'constants': constants,
            'opses': opses,
            "ans": pair['ans']
        }
        train_pairs.append(train_dict)

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    for pair in pairs_tested:
        num_stack = []
        for word in pair["out_seq"]:  # out_seq
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word: # 非符号，即word为数字
                flag_not = False
                for i, j in enumerate(pair["nums"]): # nums
                    if j == word:
                        temp_num.append(i) # 在等式的位置信息

            if not flag_not and len(temp_num) != 0:# 数字在数字列表中
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                # 数字不在数字列表中，则生成数字列表长度的位置信息，
                # 生成时根据解码器的概率选一个， 参见generate_tree_input
                num_stack.append([_ for _ in range(len(pair["nums"]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair["input_seq"])
        pos_cell = indexes_from_sentence(pos_lang, pair["pos_seq"])
        output_cell = indexes_from_sentence(output_lang, pair["out_seq"], tree)

        constants = []
        for constant in pair["constants"]:
            if constant in output_lang.word2index.keys():
                constants.append(output_lang.word2index[constant])

        opses = []
        for ops in pair["ops"]:
            if ops in ['[',']','(',')'] and tree:
                continue
            else:
                if ops == ';':
                    opses.append(output_lang.word2index['SEP'])
                else:
                    opses.append(output_lang.word2index[ops])

        vars = []
        for var in pair["unks"]:
            if var in output_lang.word2index.keys():
                vars.append(output_lang.word2index[var])

        # id type input_seq, input_len, pos_seq, pos_len, output_len, output_len, nums, num_pos, num_stack,
        # sni_nums, sni_num_pos, unks, constants, ops, ans
        # test_pairs.append((pair[0], pair[1], input_cell, len(input_cell), pos_cell, len(pos_cell), output_cell, len(output_cell),
        #                     pair[5], pair[6], num_stack, pair[7], pair[8], vars, constants, opses, pair[12],))
        test_dict = {
            "id": pair['id'],
            "type": pair['type'],
            "input_cell": input_cell,
            "input_cell_len": len(input_cell),
            "output_cell": output_cell,
            "output_cell_len": len(output_cell),
            "pos_cell": pos_cell,
            "pos_cell_len": len(pos_cell),
            "nums": pair['nums'],
            "num_pos": pair['num_pos'],
            "num_stack": num_stack,
            "sni_nums": pair['sni_nums'],
            "sni_num_pos": pair['sni_num_pos'],
            'vars': vars,
            'constants': constants,
            'opses': opses,
            "ans": pair['ans']
        }
        test_pairs.append(test_dict)

    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, pos_lang, train_pairs, test_pairs


# prepare the batches
# pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack, id, type, pos_seq, pos_len)
def prepare_train_batch(pairs_to_batch, batch_size, inlang_pad_token=0, outlang_pad_token=0, poslang_pad_token=0):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    ans_batches = []
    id_batches = []
    type_batches = []
    pos_batches = []
    pos_lengths = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp["input_cell_len"], reverse=True)
        input_length = []
        output_length = []
        pos_length = []
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack, id, type,pos_seq, pos_len)
        for pair in batch:
            input_length.append(pair["input_cell_len"])
            output_length.append(pair["output_cell_len"])
            pos_length.append(pair['pos_cell_len'])

        input_lengths.append(input_length)
        output_lengths.append(output_length)
        pos_lengths.append(pos_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        pos_len_max = max(pos_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        ans_batch = []
        id_batch = []
        type_batch = []
        pos_batch = []

        for pair in batch:
            # input_seq, input_seq_length, output_seq, output_seq_length, num, num_pos, ans, num_stack, id, type, pos_seq, pos_seq_length
            num_batch.append(pair['nums'])
            input_batch.append(pad_seq(pair['input_cell'], pair['input_cell_len'], input_len_max, pad_token=inlang_pad_token))
            output_batch.append(pad_seq(pair['output_cell'], pair['output_cell_len'], output_len_max, pad_token=outlang_pad_token))
            num_stack_batch.append(pair["num_stack"])
            num_pos_batch.append(pair['num_pos'])
            num_size_batch.append(len(pair['num_pos']))
            ans_batch.append(pair['ans'])
            id_batch.append(pair['id'])
            type_batch.append(pair['type'])
            pos_batch.append(pad_seq(pair['pos_cell'], pair['pos_cell_len'], pos_len_max, pad_token=poslang_pad_token))

        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        ans_batches.append(ans_batch)
        id_batches.append(id_batch)
        type_batches.append(type_batch)
        pos_batches.append(pos_batch)

    batches_dict = {
        "id_batches": id_batches,
        "type_batches": type_batches,
        "input_batches": input_batches,
        "input_lengths": input_lengths,
        "pos_batches": pos_batches,
        "pos_lengths": pos_lengths,
        "output_batches": output_batches,
        "output_lengths": output_lengths,
        "nums_batches": nums_batches,
        "num_stack_batches": num_stack_batches,
        "num_pos_batches": num_pos_batches,
        "num_size_batches": num_size_batches,
        "ans_batches": ans_batches,
    }
    # return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
    #        num_pos_batches, num_size_batches, ans_batches, id_batches, type_batches, pos_batches, pos_lengths
    return batches_dict


#pairs: (id type input_seq, input_len, pos_seq, pos_len, output_seq, output_len, nums, num_pos, num_stack, sni_nums, sni_num_pos, unks, constants, ops, ans）
def prepare_train_batch_all(pairs_to_batch, batch_size, inlang_pad_token=0, outlang_pad_token=0, poslang_pad_token=0,  shuffle=True):
    pairs = copy.deepcopy(pairs_to_batch)
    if shuffle: # for train
        random.shuffle(pairs)  # shuffle the pairs

    id_batches = []
    type_batches = []
    input_batches = []
    input_length_batches = []
    pos_batches = []
    pos_length_batches = []
    output_batches = []
    output_length_batches = []
    nums_batches = []
    num_pos_batches = []
    num_size_batches = []
    num_stack_batches = [] # save the num stack which
    sni_nums_batches = []
    sni_num_pos_batches = []
    sni_num_size_batches = []
    vars_batches = []
    var_size_batches = []
    constants_batches = []
    constant_size_batches = []
    opses_batches = []
    ops_size_batches = []
    ans_batches = []

    batches = []
    pos = 0
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp['input_len'], reverse=True)
        input_length = []
        output_length = []
        pos_length = []
        # pairs: (id type input_seq, input_len, pos_seq, pos_len, output_seq, output_len, nums, num_pos, num_stack, sni_nums, sni_num_pos, unks, constants, ops, ans)
        # for id, type,input_seq, input_len, pos_seq, pos_len, output_seq, output_len, nums, num_pos, num_stack, sni_nums, sni_num_pos, unks, constants, ops, ans in batch:
        for pair in batch:
            input_length.append(pair['input_cell_len'])
            pos_length.append(pair['pos_cell_len'])
            output_length.append(pair['output_cell_len'])

        input_length_batches.append(input_length)
        pos_length_batches.append(pos_length)
        output_length_batches.append(output_length)

        input_len_max = input_length[0]
        output_len_max = max(output_length)
        pos_len_max = max(pos_length)
        id_batch = []
        type_batch = []
        input_batch = []
        pos_batch = []
        output_batch = []
        nums_batch = []
        num_pos_batch = []
        num_size_batch = []
        num_stack_batch = [] # save the num stack which
        sni_nums_batch = []
        sni_num_pos_batch = []
        sni_num_size_batch = []
        vars_batch = []
        var_size_batch = []
        constants_batch = []
        constant_size_batch = []
        opses_batch = []
        ops_size_batch = []
        ans_batch = []

        # for id, type,input_seq, input_len, pos_seq, pos_len, output_seq, output_len, nums, num_pos, num_stack, sni_nums, sni_num_pos, unks, constants, ops, ans in batch:
        for pair in batch:
            id_batch.append(pair['id'])
            type_batch.append(pair['type'])
            nums_batch.append(pair['nums'])
            sni_nums_batch.append(pair['sni_nums'])
            input_batch.append(pad_seq(pair['input_cell'], pair['input_cell_len'], input_len_max, pad_token=inlang_pad_token))
            pos_batch.append(pad_seq(pair['pos_cell_len'], pair['pos_cell_len'], pos_len_max, pad_token=poslang_pad_token))
            output_batch.append(pad_seq(pair['output_cell'], pair['output_cell_len'], output_len_max, pad_token=outlang_pad_token))

            num_stack_batch.append(pair['num_stack'])
            num_pos_batch.append(pair['num_pos'])
            num_size_batch.append(len(pair['num_pos']))
            sni_num_pos_batch.append(pair['sni_num_pos'])
            sni_num_size_batch.append(len(pair['sni_num_pos']))
            vars_batch.append(pair['unks'])
            var_size_batch.append(len(pair['unks']))
            constants_batch.append(pair['constants'])
            constant_size_batch.append(len(pair['constants']))
            opses_batch.append(pair['opses'])
            ops_size_batch.append(len(pair['opses']))
            ans_batch.append(pair['ans'])

        id_batches.append(id_batch)
        type_batches.append(type_batch)
        input_batches.append(input_batch)
        pos_batches.append(pos_batch)
        output_batches.append(output_batch)
        nums_batches.append(nums_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        sni_nums_batches.append(sni_nums_batch)
        sni_num_pos_batches.append(sni_num_pos_batch)
        sni_num_size_batches.append(sni_num_size_batch)
        vars_batches.append(vars_batch)
        var_size_batches.append(var_size_batch)
        constants_batches.append(constants_batch)
        constant_size_batches.append(constant_size_batch)
        opses_batches.append(opses_batch)
        ops_size_batches.append(ops_size_batch)
        ans_batches.append(ans_batch)

    batches_dict = {
        "id_batches": id_batches,
        "type_batches": type_batches,
        "input_batches": input_batches,
        "input_length_batches": input_length_batches,
        "pos_batches": pos_batches,
        "pos_length_batches": pos_length_batches,
        "output_batches": output_batches,
        "output_length_batches": output_length_batches,
        "nums_batches": nums_batches,
        "num_pos_batches": num_pos_batches,
        "num_size_batches": num_size_batches,
        "num_stack_batches": num_stack_batches,
        "sni_nums_batches": sni_nums_batches,
        "sni_num_pos_batches": sni_num_pos_batches,
        "sni_num_size_batches": sni_num_size_batches,
        "vars_batches": vars_batches,
        "var_size_batches": var_size_batches,
        "constants_batches": constants_batches,
        "constant_size_batches": constant_size_batches,
        "opses_batches": opses_batches,
        "opses_size_batches": ops_size_batches,
        "ans_batches": ans_batches
    }
    return batches_dict
    # return id_batches, type_batches, input_batches, input_length_batches, pos_batches, pos_length_batches, \
    #        output_batches, output_length_batches, nums_batches, num_pos_batches, num_size_batches, num_stack_batches, \
    #        sni_nums_batches, sni_num_pos_batches, sni_num_size_batches, vars_batches, var_size_batches, \
    #        constants_batches, constant_size_batches, opses_batches, ops_size_batches, ans_batches


def prepare_train_batch_for_dual(pairs_to_batch, batch_size, inlang_pad_token=0, outlang_pad_token=0, poslang_pad_token=0):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    input_lm_prob_batches = []
    eq_lm_prob_batches = []
    ans_batches = []
    id_batches = []
    type_batches = []
    pos_batches = []
    pos_lengths = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        pos_length = []
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, input_lm_prob, eq_lm_prob, ans, num_stack, id, type, pos_seq, pos_len)
        # for _, i, _, j, _, _, _, _, _, _, _, _,_,k in batch:
        #     input_length.append(i)
        #     output_length.append(j)
        #     pos_length.append(k)
        for pair in batch:
            input_length.append(pair['input_cell_len'])
            pos_length.append(pair['pos_cell_len'])
            output_length.append(pair['output_cell_len'])


        input_lengths.append(input_length)
        output_lengths.append(output_length)
        pos_lengths.append(pos_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        pos_len_max = max(pos_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        input_lm_prob_batch = []
        eq_lm_prob_batch = []
        ans_batch = []
        id_batch = []
        type_batch = []
        pos_batch = []

        # for input_seq, input_seq_length, output_seq, output_seq_length, num, num_pos, input_lm_prob, eq_lm_prob, ans, num_stack, id, type, pos_seq, pos_seq_length in batch:
        for pair in batch:
            num_batch.append(pair['num'])
            input_batch.append(pad_seq(pair['input_cell'], pair['input_cell_len'], input_len_max, pad_token=inlang_pad_token))
            output_batch.append(pad_seq(pair['output_cell'], pair['output_cell_len'], output_len_max, pad_token=outlang_pad_token))
            num_stack_batch.append(pair['num_stack'])
            num_pos_batch.append(pair['num_pos'])
            num_size_batch.append(len(pair['num_pos']))
            ans_batch.append(pair['ans'])
            input_lm_prob_batch.append(pair['input_lm_prob'])
            eq_lm_prob_batch.append(pair['eq_lm_prob'])
            id_batch.append(pair['id'])
            type_batch.append(pair['type'])
            pos_batch.append(pad_seq(pair['pos_cell'], pair['pos_cell_len'], pos_len_max, pad_token=poslang_pad_token))

        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        input_lm_prob_batches.append(input_lm_prob_batch)
        eq_lm_prob_batches.append(eq_lm_prob_batch)
        ans_batches.append(ans_batch)
        id_batches.append(id_batch)
        type_batches.append(type_batch)
        pos_batches.append(pos_batch)

    batches_dict = {
        "id_batches": id_batches,
        "type_batches": type_batches,
        "input_batches": input_batches,
        "input_length_batches": input_lengths,
        "pos_batches": pos_batches,
        "pos_length_batches": pos_lengths,
        "output_batches": output_batches,
        "output_length_batches": output_lengths,
        "nums_batches": nums_batches,
        "num_pos_batches": num_pos_batches,
        "num_size_batches": num_size_batches,
        "num_stack_batches": num_stack_batches,
        "input_lm_prob_batches": input_lm_prob_batches,
        "eq_lm_prob_batches": eq_lm_prob_batches,
        "ans_batches": ans_batches
    }
    return batches_dict

    # return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
    #        num_pos_batches, num_size_batches, ans_batches, input_lm_prob_batches, eq_lm_prob_batches, \
    #        id_batches, type_batches, pos_batches, pos_lengths


def prepare_train_batch_all_for_dual(pairs_to_batch, batch_size, inlang_pad_token=0, outlang_pad_token=0, poslang_pad_token=0,  shuffle=True):
    pairs = copy.deepcopy(pairs_to_batch)
    if shuffle: # for train
        random.shuffle(pairs)  # shuffle the pairs

    id_batches = []
    type_batches = []
    input_batches = []
    input_length_batches = []
    pos_batches = []
    pos_length_batches = []
    output_batches = []
    output_length_batches = []
    nums_batches = []
    num_pos_batches = []
    num_size_batches = []
    num_stack_batches = [] # save the num stack which
    sni_nums_batches = []
    sni_num_pos_batches = []
    sni_num_size_batches = []
    vars_batches = []
    var_size_batches = []
    constants_batches = []
    constant_size_batches = []
    opses_batches = []
    ops_size_batches = []
    input_lm_prob_batches = []
    eq_lm_prob_batches = []
    ans_batches = []


    batches = []
    pos = 0
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp["input_cell_len"], reverse=True)
        input_length = []
        output_length = []
        pos_length = []
        # pairs: (id, type,input_seq, input_len, pos_seq, pos_len, output_seq, output_len, nums, num_pos, num_stack, sni_nums, sni_num_pos, unks, constants, ops, input_lm_prob, eq_lm_prob, ans)
        # for id, type,input_seq, input_len, pos_seq, pos_len, output_seq, output_len, nums, num_pos, num_stack, sni_nums, sni_num_pos, unks, constants, ops, input_lm_prob, eq_lm_prob, ans in batch:
        #     input_length.append(input_len)
        #     pos_length.append(pos_len)
        #     output_length.append(output_len)
        for pair in batch:
            input_length.append(pair['input_cell_len'])
            pos_length.append(pair['pos_cell_len'])
            output_length.append(pair['output_cell_len'])


        input_length_batches.append(input_length)
        pos_length_batches.append(pos_length)
        output_length_batches.append(output_length)

        input_len_max = input_length[0]
        output_len_max = max(output_length)
        pos_len_max = max(pos_length)
        id_batch = []
        type_batch = []
        input_batch = []
        pos_batch = []
        output_batch = []
        nums_batch = []
        num_pos_batch = []
        num_size_batch = []
        num_stack_batch = [] # save the num stack which
        sni_nums_batch = []
        sni_num_pos_batch = []
        sni_num_size_batch = []
        vars_batch = []
        var_size_batch = []
        constants_batch = []
        constant_size_batch = []
        opses_batch = []
        ops_size_batch = []
        input_lm_prob_batch = []
        eq_lm_prob_batch = []
        ans_batch = []

        # for id, type, input_seq, input_len, pos_seq, pos_len, output_seq, output_len, nums, num_pos, num_stack, sni_nums, sni_num_pos, unks, constants, ops, input_lm_prob, eq_lm_prob, ans in batch:
        for pair in batch:
            id_batch.append(pair['id'])
            type_batch.append(pair['type'])
            nums_batch.append(pair['nums'])
            sni_nums_batch.append(pair['sni_nums'])
            input_batch.append(pad_seq(pair['input_cell'], pair['input_cell_len'], input_len_max, pad_token=inlang_pad_token))
            pos_batch.append(pad_seq(pair['pos_cell'], pair['pos_cell_len'], pos_len_max, pad_token=poslang_pad_token))
            output_batch.append(pad_seq(pair['output_cell'], pair['output_cell_len'], output_len_max, pad_token=outlang_pad_token))

            num_stack_batch.append(pair['num_stack'])
            num_pos_batch.append(pair['num_pos'])
            num_size_batch.append(len(pair['num_pos']))
            sni_num_pos_batch.append(pair['sni_num_pos'])
            sni_num_size_batch.append(len(pair['sni_num_pos']))
            vars_batch.append(pair['vars'])
            var_size_batch.append(len(pair['vars']))
            constants_batch.append(pair['constants'])
            constant_size_batch.append(len(pair['constants']))
            opses_batch.append(pair['opses'])
            ops_size_batch.append(len(pair['opses']))
            input_lm_prob_batch.append(pair['input_lm_prob'])
            eq_lm_prob_batch.append(pair['eq_lm_prob'])
            ans_batch.append(pair['ans'])

        id_batches.append(id_batch)
        type_batches.append(type_batch)
        input_batches.append(input_batch)
        pos_batches.append(pos_batch)
        output_batches.append(output_batch)
        nums_batches.append(nums_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        sni_nums_batches.append(sni_nums_batch)
        sni_num_pos_batches.append(sni_num_pos_batch)
        sni_num_size_batches.append(sni_num_size_batch)
        vars_batches.append(vars_batch)
        var_size_batches.append(var_size_batch)
        constants_batches.append(constants_batch)
        constant_size_batches.append(constant_size_batch)
        opses_batches.append(opses_batch)
        ops_size_batches.append(ops_size_batch)
        input_lm_prob_batches.append(input_lm_prob_batch)
        eq_lm_prob_batches.append(eq_lm_prob_batch)
        ans_batches.append(ans_batch)

    batches_dict = {
        "id_batches": id_batches,
        "type_batches": type_batches,
        "input_batches": input_batches,
        "input_length_batches": input_length_batches,
        "pos_batches": pos_batches,
        "pos_length_batches": pos_length_batches,
        "output_batches": output_batches,
        "output_length_batches": output_length_batches,
        "nums_batches": nums_batches,
        "num_pos_batches": num_pos_batches,
        "num_size_batches": num_size_batches,
        "num_stack_batches": num_stack_batches,
        "sni_nums_batches": sni_nums_batches,
        "sni_num_pos_batches": sni_num_pos_batches,
        "sni_num_size_batches": sni_num_size_batches,
        "vars_batches": vars_batches,
        "var_size_batches": var_size_batches,
        "constants_batches": constants_batches,
        "constant_size_batches": constant_size_batches,
        "opses_batches": opses_batches,
        "opses_size_batches": ops_size_batches,
        "input_lm_prob_batches": input_lm_prob_batches,
        "eq_lm_prob_batches": eq_lm_prob_batches,
        "ans_batches": ans_batches
    }
    return batches_dict

    # return id_batches, type_batches, input_batches, input_length_batches, pos_batches, pos_length_batches, \
    #        output_batches, output_length_batches, nums_batches, num_pos_batches, num_size_batches, num_stack_batches, \
    #        sni_nums_batches, sni_num_pos_batches, sni_num_size_batches, vars_batches, var_size_batches, \
    #        constants_batches, constant_size_batches, opses_batches, ops_size_batches, input_lm_prob_batches,\
    #        eq_lm_prob_batches, ans_batches

# prepare the batches
def prepare_test_batch(pairs_to_batch, batch_size, inlang_pad_token=0, outlang_pad_token=0, poslang_pad_token=0):
    pairs = copy.deepcopy(pairs_to_batch)
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    ans_batches = []
    id_batches = []
    type_batches = []
    pos_batches = []
    pos_lengths = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp["input_cell_len"], reverse=True)
        input_length = []
        output_length = []
        pos_length = []
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack, id, type,pos_seq, pos_len)
        # for _, i, _, j, _, _, _, _, _, _, _, k in batch:
        #     input_length.append(i)
        #     output_length.append(j)
        #     pos_length.append(k)
        for pair in batch:
            input_length.append(pair['input_cell_len'])
            pos_length.append(pair['pos_cell_len'])
            output_length.append(pair['output_cell_len'])

        input_lengths.append(input_length)
        output_lengths.append(output_length)
        pos_lengths.append(pos_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        pos_len_max = max(pos_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        ans_batch = []
        id_batch = []
        type_batch = []
        pos_batch = []

        # for input_seq, input_seq_length, output_seq, output_seq_length, num, num_pos, ans, num_stack, id, type, pos_seq, pos_seq_length in batch:
        for pair in batch:
            num_batch.append(pair['num'])
            input_batch.append(pad_seq(pair['input_cell'], pair['input_cell_len'], input_len_max, pad_token=inlang_pad_token))
            output_batch.append(pad_seq(pair['output_cell'], pair['output_cell_len'], output_len_max, pad_token=outlang_pad_token))
            num_stack_batch.append(pair['num_stack'])
            num_pos_batch.append(pair['num_pos'])
            num_size_batch.append(len(pair['num_pos']))
            ans_batch.append(pair['ans'])
            id_batch.append(pair['id'])
            type_batch.append(pair['type'])
            pos_batch.append(pad_seq(pair['pos_cell'], pair['pos_cell_len'], pos_len_max, pad_token=poslang_pad_token))


        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        ans_batches.append(ans_batch)
        id_batches.append(id_batch)
        type_batches.append(type_batch)
        pos_batches.append(pos_batch)

    batches_dict = {
        "id_batches": id_batches,
        "type_batches": type_batches,
        "input_batches": input_batches,
        "input_lengths": input_lengths,
        "pos_batches": pos_batches,
        "pos_lengths": pos_lengths,
        "output_batches": output_batches,
        "output_lengths": output_lengths,
        "nums_batches": nums_batches,
        "num_stack_batches": num_stack_batches,
        "num_pos_batches": num_pos_batches,
        "num_size_batches": num_size_batches,
        "ans_batches": ans_batches,
    }

    return batches_dict

    # return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
    #        num_pos_batches, num_size_batches, ans_batches, id_batches, type_batches, pos_batches, pos_lengths
