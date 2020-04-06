##
# 统计所有keyword 的数量，并生成
#
#
# ##


import pandas as pd
from util import utiler
import re
import progressbar
number_sub= re.compile('[0-9]*')

def add(map, word):
    if len(word) < 3 or word.isdigit():
        return map
    if word in map.keys():
        map[j] = map[word] + 1
    else:
        map[word] = 1
    return map

tag_path = 'D:/Github/BERT-Keyword-Extractor/GKB_doc/tag_count_all.csv'
datapath = 'D:\Github\BERT-Keyword-Extractor\GKB_doc\GRB_training_text_en_all.xlsx'
cop = re.compile("[^ !#$%&\'()*+;?^_`{|}~^a-z^A-Z^0-9]")
html_sub = re.compile('<\\/?.+?\\/?>')# html标签
# number_sub= re.compile('[0-9\#\$\%&]*')

data = pd.DataFrame(pd.read_excel(datapath))
# data['filtered_result']=''
datamapt = {'Name': 'Zara'}
datamapt.clear()
with progressbar.ProgressBar(max_value=100) as bar:
    for i in range(len(data)):
        bar.update(round(i / len(data), 6) * 100)
        if data.values[i, 6] == '' or data.values[i, 5] == '' or data.values[i, 7] == '':
            # data['filtered_result'][i] = ''
            continue
            # if data.values[i, 7]!=data.values[i, 7] or data.values[i, 6]!=data.values[i, 6] or data.values[i, 5]!=data.values[i, 5]: #nan
        if data.values[i, 7] != data.values[i, 7]:
            # data['filtered_result'][i] = ''
            continue
        en_tag = data.values[i, 7]
        en_tag = en_tag.replace(',', '').replace('，', '').replace('&lt', '').replace('&gt', '').replace('&quot','').replace('&nbsp', '').replace('/', ' ')
        en_tag = cop.sub("", en_tag)
        en_tag = html_sub.sub('', en_tag)
        en_tag = en_tag.replace('keywords:', '').replace('.', '')
        # data['filtered_result'][i] = en_tag
        for j in en_tag.split(';'):
            #j = number_sub.sub('', j)
            flag_has_letter=False
            if len(j) < 3:
                continue
            if '(' in j and ')' not in j:
                continue
            if ')' in j and '(' not in j:
                continue
            if j[0] in '\\/=-!#$%&\'()*+;?^_`{|}~':
                continue
            for key in j:
                if key in 'abcdefghjklmnopqrstuvwxyz':
                    flag_has_letter=True
                    break
            if not flag_has_letter:
                continue

            if '(' in j and ')' in j:
                ta=0
                while '(' in j and ')' in j and ta<10: #一个句子中有可能有两个括号
                    short = j[j.find('(') + 1:j.find(')')]
                    j = j[0: j.find('(')] + j[j.find(')') + 1: len(j)]
                    datamapt = add(datamapt, short)
                    ta=ta+1
                datamapt = add(datamapt, j.strip())
                continue
            j = j.strip()
            datamapt = add(datamapt, j)
# data.to_excel('D:\Github\BERT-Keyword-Extractor\GKB_doc\GRB_training_text_en_all_filltter.xlsx', index=False)
datamapt = sorted(datamapt.items(), key=lambda item: item[1], reverse=True)
for key in datamapt:
    tt = key[0] + ',' + str(key[1]) + ',' + str(len(key[0].split(' ')))
    utiler.save_txt(tt, tag_path)
