import random
import json
import copy
import re
from text_num_utils import isfloat, en_number_pattern, en_text2num, en_fraction_pattern


chs_arabic_map = {u'零':0, u'一':1, u'二':2, u'三':3, u'四':4,
                  u'五':5, u'六':6, u'七':7, u'八':8, u'九':9,
                  u'十':10, u'百':100, u'千': 10 ** 3, u'万':10 ** 4,
                  u'〇':0, u'壹':1, u'贰':2, u'叁':3, u'肆':4,
                  u'伍':5, u'陆':6, u'柒':7, u'捌':8, u'玖':9,
                  u'拾':10, u'佰':100, u'仟':10 ** 3, u'萬':10 ** 4,
                  u'亿':10 ** 8, u'億':10 ** 8, u'幺': 1,
                  u'０':0, u'１':1, u'２':2, u'３':3, u'４':4,
                  u'５':5, u'６':6, u'７':7, u'８':8, u'９':9,u'两':2,
                  '0':0, '1':1, '2':2, '3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,} # u'双':2,


def convert_chinese_digits_to_arabic(chinese_digits, encoding="utf-8"):
    if not isinstance(chinese_digits, str):
        chinese_digits = chinese_digits.decode(encoding)

    result = 0
    tmp = 0
    hnd_mln = 0
    point = False
    after_point = 0
    for count in range(len(chinese_digits)):
        curr_char = chinese_digits[count]
        if curr_char == '.' or curr_char == u'点':
            point = True
            after_point = 10
            continue

        curr_digit = chs_arabic_map.get(curr_char, None)
        # meet 「亿」 or 「億」
        if curr_digit == 10 ** 8:
            result = result + tmp
            result = result * curr_digit
            # get result before 「亿」 and store it into hnd_mln
            # reset `result`
            hnd_mln = hnd_mln * 10 ** 8 + result
            result = 0
            tmp = 0
            point = False
        # meet 「万」 or 「萬」
        elif curr_digit == 10 ** 4:
            result = result + tmp
            result = result * curr_digit
            tmp = 0
            point = False
        # meet 「十」, 「百」, 「千」 or their traditional version
        elif curr_digit >= 10:
            tmp = 1 if tmp == 0 else tmp
            result = result + curr_digit * tmp
            tmp = 0
            point = False
        # meet single digit
        elif curr_digit is not None:
            if not point:
                tmp = tmp * 10 + curr_digit
            else:
                tmp = tmp + curr_digit / after_point
                after_point *= 10
        else:
            return result
    result = result + tmp
    result = result + hnd_mln
    return result


def replace_en_number_with_digits(text, str_bool=False):
    text = text.replace('$', '$ ') \
        .replace('/', ' / ') \
        .replace('a half', '1.5')
    text = text.replace('twice', '2 times')
    text = text.replace('double', '2 times')

    # detect case: num1    num2   /   num3  --> num1 + (num2/num3)
    for match in re.finditer('([0-9]+) ([0-9]+) */ *([0-9]+)', text):
        frac = int(match.group(1)) + int(match.group(2)) / int(match.group(3))
        text = text.replace(match.group(), str(frac))

    # detect case: num1   /   num2  --> num1/num2
    for match in re.finditer(r'\(([0-9]+) */ *([0-9]+)\)', text):
        frac = int(match.group(1)) / int(match.group(2))
        text = text.replace(match.group(), str(frac))

    # detect case: x.x%
    for match in re.finditer(r'([0-9]+\.)?[0-9]+%', text):
        percent_text = match.group()
        float_text = str(float(percent_text[:-1]) / 100)
        text = text.replace(percent_text,
                            float_text)

    # detect case: 100,000
    for match in re.finditer('\\d{1,3}(,\\d{3})+', text):
        match_text = match.group()
        text = text.replace(match_text,
                            match_text.replace(',', ''))

    # detect fraction case： one third
    for num_text in en_fraction_pattern.finditer(text):
        print(num_text.group())
        print(str(en_text2num(num_text.group(),str_bool=str_bool)))
        text = text.replace(num_text.group(),
                            str(en_text2num(num_text.group(),str_bool=str_bool)))

    # replace number with digits
    for num_text in en_number_pattern.finditer(text):
        text = text.replace(num_text.group(),
                            str(en_text2num(num_text.group(),str_bool=str_bool)))

    text = re.sub(r'(-?\d+.\d+|\d+)', r' \1 ', text)
    text = re.sub(r' +', ' ', text)
    return text


# def replace_cn_number_with_digits(chinese_digits, encoding="utf-8"):
#     if isinstance (chinese_digits, str):
#         chinese_digits = chinese_digits.decode(encoding)
#
#     result = 0
#     tmp = 0
#     hnd_mln = 0
#     for count in range(len(chinese_digits)):
#         curr_char  = chinese_digits[count]
#         curr_digit = chs_arabic_map.get(curr_char, None)
#         # meet 「亿」 or 「億」
#         if curr_digit == 10 ** 8:
#             result = result + tmp
#             result = result * curr_digit
#             # get result before 「亿」 and store it into hnd_mln
#             # reset `result`
#             hnd_mln = hnd_mln * 10 ** 8 + result
#             result = 0
#             tmp = 0
#         # meet 「万」 or 「萬」
#         elif curr_digit == 10 ** 4:
#             result = result + tmp
#             result = result * curr_digit
#             tmp = 0
#         # meet 「十」, 「百」, 「千」 or their traditional version
#         elif curr_digit >= 10:
#             tmp = 1 if tmp == 0 else tmp
#             result = result + curr_digit * tmp
#             tmp = 0
#         # meet single digit
#         elif curr_digit is not None:
#             tmp = tmp * 10 + curr_digit
#         else:
#             return result
#     result = result + tmp
#     result = result + hnd_mln
#     return result

# remove the superfluous brackets
def remove_brackets(x):
    y = x
    if x[0] == "(" and x[-1] == ")":
        x = x[1:-1]
        flag = True
        count = 0
        for s in x:
            if s == ")":
                count -= 1
                if count < 0:
                    flag = False
                    break
            elif s == "(":
                count += 1
        if flag:
            return x
    return y


def check_bracket(x, english=False):
    if english:
        for idx, s in enumerate(x):
            if s == '[':
                x[idx] = '('
            elif s == '}':
                x[idx] = ')'
        s = x[0]
        idx = 0
        if s == "(":
            flag = 1
            temp_idx = idx + 1
            while flag > 0 and temp_idx < len(x):
                if x[temp_idx] == ")":
                    flag -= 1
                elif x[temp_idx] == "(":
                    flag += 1
                temp_idx += 1
            if temp_idx == len(x):
                x = x[idx + 1:temp_idx - 1]
            elif x[temp_idx] != "*" and x[temp_idx] != "/":
                x = x[idx + 1:temp_idx - 1] + x[temp_idx:]
        while True:
            y = len(x)
            for idx, s in enumerate(x):
                if s == "+" and idx + 1 < len(x) and x[idx + 1] == "(":
                    flag = 1
                    temp_idx = idx + 2
                    while flag > 0 and temp_idx < len(x):
                        if x[temp_idx] == ")":
                            flag -= 1
                        elif x[temp_idx] == "(":
                            flag += 1
                        temp_idx += 1
                    if temp_idx == len(x):
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1]
                        break
                    elif x[temp_idx] != "*" and x[temp_idx] != "/":
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1] + x[temp_idx:]
                        break
            if y == len(x):
                break
        return x

    lx = len(x)
    for idx, s in enumerate(x):
        if s == "[":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == "]":
                    flag_b += 1
                elif x[temp_idx] == "[":
                    flag_b -= 1
                if x[temp_idx] == "(" or x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == "]" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "("
                x[temp_idx] = ")"
                continue
        if s == "(":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == ")":
                    flag_b += 1
                elif x[temp_idx] == "(":
                    flag_b -= 1
                if x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == ")" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "["
                x[temp_idx] = "]"
    return x


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    # if "SOS" in lang.index2word and not tree:
    #     res.append(lang.word2index["SOS"])
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res


def indexes_from_constants(lang, word_list):
    res = []
    # if "SOS" in lang.index2word and not tree:
    #     res.append(lang.word2index["SOS"])
    for word in word_list:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
    return res


# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length, pad_token=0):
    seq += [pad_token for _ in range(max_length - seq_len)]
    return seq


# 用于获取等式中没有出现在输出字典中的数字
def get_num_stack(eq, output_lang, num_pos):
    num_stack = []
    for word in eq:
        temp_num = []
        flag_not = True
        if word not in output_lang.index2word:
            flag_not = False
            for i, j in enumerate(num_pos):
                if j == word:
                    temp_num.append(i)
        if not flag_not and len(temp_num) != 0:  # 数字/符号不在词表中，但在等式中出现
            num_stack.append(temp_num)
        if not flag_not and len(temp_num) == 0:  # 数字/符号不在词表中，且不在等式中出现
            num_stack.append([_ for _ in range(len(num_pos))])
    num_stack.reverse()
    return num_stack


# 将模型输出的表达式(id表示)转换为真正human可读的表达式
def convert_expression_list(expression, output_lang, num_list, num_stack=None):
    max_index = output_lang.n_words
    res = []
    for i in expression:
        # if i == 0:
        #     return res
        if i < max_index - 1:
            idx = output_lang.index2word[i]
            if idx[0] == "N":
                if int(idx[1:]) >= len(num_list):
                    return None
                res.append(num_list[int(idx[1:])])
            else:
                res.append(idx)
        else:
            pos_list = num_stack.pop()
            c = num_list[pos_list[0]]
            res.append(c)
    return res






















