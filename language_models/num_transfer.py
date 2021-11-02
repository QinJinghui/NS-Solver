from sympy import *
import re

def transfer_cm17k_num(data_list):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    max_id = 0
    for d in data_list:
        nums = []
        input_seq = []
        seg = d["original_text"].strip().split()
        if len(seg) > 200:
            continue
        equations = d["equation"].replace('[','(').replace(']',')')

        for s in seg:
            pos = re.search(pattern, s)  # 搜索每个词的数字位置
            if pos and pos.start() == 0:
                nums.append(s[pos.start():pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        # if len(input_seq) > 384:
        #     continue

        if copy_nums < len(nums):
            # if len(nums) > 20:
            #     continue
            copy_nums = len(nums)
            # max_id = d['id']

        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True) # 从大到小排序

        # print(nums)
        # print(nums_fraction)
        float_nums = []
        for num in nums:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums.append(str(float(eval(num.strip()))))
            elif '%' in num:
                float_nums.append(str(float(round(eval(num[:-1].strip()) / 100, 3))))
            elif len(num) > 1 and num[0] == '0':
                float_nums.append(str(float(eval(num[1:].strip()))))
            else:
                float_nums.append(str(float(eval(num.strip()))))

        float_nums_fraction = []
        for num in nums_fraction:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums_fraction.append(str(float(eval(num.strip()))))
            elif '%' in num:
                # float_nums.append(str(float(round(eval(num[:-1].strip()) / 100, 3))))
                float_nums_fraction.append(str(float(round(eval(num[:-1].strip()) / 100, 3))))
            else:
                float_nums_fraction.append(str(float(eval(num.strip()))))
        # print(float_nums)
        # print(float_nums_fraction)
        nums = float_nums
        nums_fraction = float_nums_fraction

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N" + str(nums.index(n)))
                    # elif nums.count(n) > 1:
                    #     # 多个的时候默认使用第一个index代替
                    #     res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res

            pos_st = re.search("\d+\.\d+%?|\d+%?", st) # 带百分号的数字数
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        new_out_seq = []
        for seq in out_seq:
            if seq == ' ' or seq == '':
                continue
            if seq == ';':
                new_out_seq.append('SEP')
                continue
            new_out_seq.append(seq)
        out_seq = new_out_seq

        idx = 0
        new_out_seq = []
        while idx < len(out_seq):
            if out_seq[idx] == 'SEP':
                new_out_seq.append(out_seq[idx])
                idx += 1
                continue
            if idx + 1 < len(out_seq):
                if out_seq[idx][0] == 'N' and (out_seq[idx+1] in 'xyz(' or  out_seq[idx+1][0].isdigit()):
                    new_out_seq.append(out_seq[idx])
                    new_out_seq.append('*')
                elif out_seq[idx][0] == ')' and out_seq[idx+1] not in '+-*/^=)SEP':
                    new_out_seq.append(out_seq[idx])
                    new_out_seq.append('*')
                else:
                    new_out_seq.append(out_seq[idx])
            else:
                new_out_seq.append(out_seq[idx])
            idx += 1
        out_seq = new_out_seq

        # print(equations)
        # print(' '.join(out_seq))
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        ans = d['ans']
        # if len(input_seq) > 256:
        #     input_seq = input_seq[:256]
        id = d['id']
        type = d['type']
        pos_seq = d["pos"].strip().split()
        # pairs.append((input_seq, out_seq, nums, num_pos, ans, id, type, pos_seq))
        data_dict = {
            "id": id,
            "type": type,
            "input_seq": input_seq,
            "out_seq": out_seq,
            "pos_seq": pos_seq,
            "nums": nums,
            "num_pos": num_pos,
            "ans": ans,
        }
        pairs.append(data_dict)

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    max_num_list_len = copy_nums
    # print(len(temp_g))
    # print(max_num_list_len)
    # print(max_id)
    print(temp_g, max_num_list_len)
    return pairs, temp_g, max_num_list_len



def transfer_cm17k_num_all(data_list):
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    max_problem_len = 0
    max_id = 0
    for d in data_list:
        nums = []
        input_seq = []
        seg = d["original_text"].strip().split()
        if len(seg) > 200:
            continue
        if len(seg) > max_problem_len:
            max_problem_len = len(seg)
        equations = ''.join(d["equation"].strip().split())

        for s in seg:
            pos = re.search(pattern, s)  # 搜索每个词的数字位置
            if pos and pos.start() == 0:
                nums.append(s[pos.start():pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        # if len(input_seq) > 384:
        #     continue

        if copy_nums < len(nums):
            # if len(nums) > 20:
            #     continue
            copy_nums = len(nums)
            # max_id = d['id']

        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True) # 从大到小排序

        # print(nums)
        # print(nums_fraction)
        float_nums = []
        for num in nums:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums.append(str(float(eval(num.strip()))))
            elif '%' in num:
                float_nums.append(str(float(round(eval(num[:-1].strip()) / 100, 3))))
            elif len(num) > 1 and num[0] == '0':
                float_nums.append(str(float(eval(num[1:].strip()))))
            else:
                float_nums.append(str(float(eval(num.strip()))))

        float_nums_fraction = []
        for num in nums_fraction:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums_fraction.append(str(float(eval(num.strip()))))
            elif '%' in num:
                # float_nums.append(str(float(round(eval(num[:-1].strip()) / 100, 3))))
                float_nums_fraction.append(str(float(round(eval(num[:-1].strip()) / 100, 3))))
            else:
                float_nums_fraction.append(str(float(eval(num.strip()))))
        # print(float_nums)
        # print(float_nums_fraction)
        nums = float_nums
        nums_fraction = float_nums_fraction

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) >= 1:
                        res.append("N" + str(nums.index(n)))
                    # elif nums.count(n) > 1:
                    #     # 多个的时候默认使用第一个index代替
                    #     res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res

            pos_st = re.search("\d+\.\d+%?|\d+%?", st) # 带百分号的数字数
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) >= 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        # if d['id'] == 31832:
        #     # out_seq = seg_and_tag(equations)
        #     print(nums)
        #     print(out_seq)
        #     exit(0)
        new_out_seq = []
        for seq in out_seq:
            if seq == ' ' or seq == '':
                continue
            if seq == ';':
                new_out_seq.append('SEP')
                continue
            new_out_seq.append(seq)
        out_seq = new_out_seq

        idx = 0
        new_out_seq = []
        while idx < len(out_seq):
            if out_seq[idx] == 'SEP':
                new_out_seq.append(out_seq[idx])
                idx += 1
                continue
            if idx + 1 < len(out_seq):
                if out_seq[idx][0] == 'N' and (out_seq[idx+1] in 'xyz(' or  out_seq[idx+1][0].isdigit()):
                    new_out_seq.append(out_seq[idx])
                    new_out_seq.append('*')
                elif out_seq[idx][0] == ')' and out_seq[idx+1] not in '+-*/^=)SEP':
                    new_out_seq.append(out_seq[idx])
                    new_out_seq.append('*')
                else:
                    new_out_seq.append(out_seq[idx])
            else:
                new_out_seq.append(out_seq[idx])
            idx += 1
        out_seq = new_out_seq

        # print(equations)
        # print(' '.join(out_seq))
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        sni_num_pos = d['sni_loc']
        sni_nums = []
        for pos, num in zip(num_pos, nums):
            if pos in sni_num_pos:
                sni_nums.append(num)
        ans = d['ans']
        # if len(input_seq) > 256:
        #     input_seq = input_seq[:256]
        id = d['id']
        type = d['type']
        pos_seq = d["pos"].strip().split()
        constants = d['constant']
        unks = d['unk']
        ops = d['ops']
        # pairs.append((id, type, input_seq, pos_seq, out_seq, nums, num_pos,
        #               sni_nums, sni_num_pos, unks, constants, ops, ans,))
        data_dict = {
            "id": id,
            "type": type,
            "input_seq": input_seq,
            "out_seq": out_seq,
            "pos_seq": pos_seq,
            "nums": nums,
            "num_pos": num_pos,
            "ans": ans,
            "sni_nums": sni_nums,
            "sni_num_pos": sni_num_pos,
            "unks": unks,
            "constants": constants,
            "ops": ops
        }
        pairs.append(data_dict)

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    max_num_list_len = copy_nums
    return pairs, temp_g, max_num_list_len, max_problem_len








