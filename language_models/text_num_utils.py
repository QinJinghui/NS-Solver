import re

en_to_9 = '(zero|one|two|three|four|five|six|seven|eight|nine)'
cn_to_9 = '(零|一|二|三|四|五|六|七|八|九)'
cn_to_09 = '(零一|零二|零三|零四|零五|零六|零七|零八|零九)'
en_xty = '(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)'
cn_xty = '(二十|三十|四十|五十|六十|七十|八十|九十)'
en_to_19 = '(ten|eleven|twelve|thirteen|fourteen|fifteen' \
           '|sixteen|seventeen|eighteen|nineteen|{})'.format(en_to_9)
cn_to_19 = '(十|一十|十一|十二|十三|十四|十五|十六|十七|十八|十九|{})'.format(cn_to_9)
cn_to_019 = '(一十|一十一|一十二|一十三|一十四|一十五|一十六|一十七|一十八|一十九|{})'.format(cn_to_09)

en_to_99 = '(({xty}[ -]{to_9})|{xty}|{to_19})'.format(
    to_19=en_to_19,
    to_9=en_to_9,
    xty=en_xty)
cn_to_99 = '(({xty}{to_9})|{xty}|{to_19})'.format(
    to_19=cn_to_19,
    to_9=cn_to_9,
    xty=cn_xty)
cn_to_099 = '(({xty}{to_9})|{xty}|{to_19})'.format(
    to_19=cn_to_019,
    to_9=cn_to_9,
    xty=cn_xty)
en_to_999 = '({to_9} hundred( (and )?{to_99})?|{to_99})'.format(
    to_9=en_to_9, to_99=en_to_99)
cn_to_999 = '({to_9}百{to_099}|{to_99})'.format(
    to_9=cn_to_9, to_099=cn_to_099, to_99=cn_to_99)
en_to_999999 = '({to_999} thousand( (and)? {to_999})?|{to_999})'.format(
    to_999=en_to_999)
# cn_to_999999 ='{to_999}千{to_999}|{to_999})'.format(
#     to_999=cn_to_999)
en_to_9x9 = '({to_999999} million( (and)? {to_999999})?|{to_999999})'.format(
    to_999999=en_to_999999)

en_to_9x12 = '({to_9x9} billion( (and)? {to_9x9})?|{to_9x9})'.format(
    to_9x9=en_to_9x9)

en_fraction = '({to_19}-(second|thirds|fourths|' \
              'fifths|sixths|sevenths|eighths|nineths|tenths|elevenths|twelfths|' \
              'thirteenths|fourteenths|fifteenths|sixteenths|seventeenths|eighteenths|' \
              'nineteenths|twentyths|third|fourth|fifth|sixth|seventh|eighth|ninth|' \
              'tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|' \
              'sixteenth|seventeenth|eighteenth|nineteenth|twentyth)|half|quarter|' \
              '{to_19} (second|thirds|fourths|' \
              'fifths|sixths|sevenths|eighths|nineths|tenths|elevenths|twelfths|' \
              'thirteenths|fourteenths|fifteenths|sixteenths|seventeenths|eighteenths|' \
              'nineteenths|twentyths |third|fourth|fifth|sixth|seventh|eighth|ninth|' \
              'tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|' \
              'sixteenth|seventeenth|eighteenth|nineteenth|twentyth))'.format(
    to_19=en_to_19)

en_numbers = '(({to_9x12} and )?{fraction}|{to_9x12})'.format(
    to_9x12=en_to_9x12, fraction=en_fraction)

en_fraction_pattern = re.compile(en_fraction)
en_number_pattern = re.compile(en_numbers)


def en_text2num(text, str_bool=False):
    """ Convert text to number.
    """
    base = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'eleven': 11,
        'twelve': 12,
        'thirteen': 13,
        'fourteen': 14,
        'fifteen': 15,
        'sixteen': 16,
        'seventeen': 17,
        'eighteen': 18,
        'nineteen': 19,
        'twenty': 20,
        'thirty': 30,
        'forty': 40,
        'fifty': 50,
        'sixty': 60,
        'seventy': 70,
        'eighty': 80,
        'ninety': 90,
        'twice': 2,
        'half': 0.5,
        'quarter': 0.25}

    scale = {
        'thousand': 1000,
        'million': 1000000,
        'billion': 1000000000}

    order = {
        'second': 2,
        'thirds': 3,
        'fourths': 4,
        'fifths': 5,
        'sixths': 6,
        'sevenths': 7,
        'eighths': 8,
        'nineths': 9,
        'tenths': 10,
        'elevenths': 11,
        'twelfths': 12,
        'thirteenths': 13,
        'fourteenths': 14,
        'fifteenths': 15,
        'sixteenths': 16,
        'seventeenths': 17,
        'eighteenths': 18,
        'nineteenths': 19,
        'twentyths': 20,
        'third': 3,
        'fourth': 4,
        'fifth': 5,
        'sixth': 6,
        'seventh': 7,
        'eighth': 8,
        'nineth': 9,
        'tenth': 10,
        'eleventh': 11,
        'twelfth': 12,
        'thirteenth': 13,
        'fourteenth': 14,
        'fifteenth': 15,
        'sixteenth': 16,
        'seventeenth': 17,
        'eighteenth': 18,
        'nineteenth': 19,
        'twentyth': 20}

    tokens = []
    for token in text.split(' '):
        if token == 'and':
            continue
        elif '-' in token:
            if token.split('-')[-1] in order:
                tokens.append(token)
            else:
                tokens += token.split('-')
        else:
            tokens.append(token)

    if str_bool:
        result = ""
    else:
        result = 0
    leading = 0
    for token in tokens:
        if token in base:
            if str_bool and token == "half":
                result += '1 / 2'
            elif str_bool and token == "quarter":
                result += '1 / 4'
            else:
                leading += base[token]
        elif token == 'hundred':
            leading *= 100
        elif token in scale:
            if str_bool:
                result += str(leading) + ' * ' + str(scale[token])
            else:
                result += leading * scale[token]
            leading = 0
        elif token in order:
            if str_bool:
                result += str(leading) + ' / ' + str(order[token])
            else:
                result += leading / order[token]
            leading = 0
        elif '-' in token:
            numerator, denominator = token.split('-')
            if str_bool:
                result += str(base[numerator]) + ' / ' + str(order[denominator])
            else:
                result += base[numerator] / order[denominator]
    if str_bool:
        if leading != 0:
            result += str(leading)
    else:
        result += leading

    return result


def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


#encoding: utf-8
chs_arabic_map = {u'零':0, u'一':1, u'二':2, u'三':3, u'四':4,
                  u'五':5, u'六':6, u'七':7, u'八':8, u'九':9,
                  u'十':10, u'百':100, u'千':10 ** 3, u'万':10 ** 4,
                  u'〇':0, u'壹':1, u'贰':2, u'叁':3, u'肆':4,
                  u'伍':5, u'陆':6, u'柒':7, u'捌':8, u'玖':9,
                  u'拾':10, u'佰':100, u'仟':10 ** 3, u'萬':10 ** 4,
                  u'亿':10 ** 8, u'億':10 ** 8, u'幺': 1,
                  u'０':0, u'１':1, u'２':2, u'３':3, u'４':4,
                  u'５':5, u'６':6, u'７':7, u'８':8, u'９':9}


def convertChineseDigitsToArabic(chinese_digits, encoding="utf-8"):
    if isinstance (chinese_digits, str):
        chinese_digits = chinese_digits.decode(encoding)

    result = 0
    tmp = 0
    hnd_mln = 0
    for count in range(len(chinese_digits)):
        curr_char = chinese_digits[count]
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
        # meet 「万」 or 「萬」
        elif curr_digit == 10 ** 4:
            result = result + tmp
            result = result * curr_digit
            tmp = 0
        # meet 「十」, 「百」, 「千」 or their traditional version
        elif curr_digit >= 10:
            tmp = 1 if tmp == 0 else tmp
            result = result + curr_digit * tmp
            tmp = 0
        # meet single digit
        elif curr_digit is not None:
            tmp = tmp * 10 + curr_digit
        else:
            return result
    result = result + tmp
    result = result + hnd_mln
    return result


