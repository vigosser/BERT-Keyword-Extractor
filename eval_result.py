##
#  评估keyword 生成的结果，分为exact match跟 partial match
#
# ##

import pandas as pd
from seqevl_util import seqeval_util
import re
cop = re.compile("[^ !#$%&\'()*+,-./:;<=>?\\^_`{|}~^a-z^A-Z^0-9]")
base_dir='D:\Github\BERT-Keyword-Extractor'
# data = pd.DataFrame(pd.read_excel(base_dir+'/GKB_doc/cn_finalresult.xlsx', encoding='utf8'))
# data = pd.DataFrame(pd.read_excel('D:\Github\BERT-Keyword-Extractor\GKB_doc\en_finalresult_remove_empty.xlsx', encoding='utf8'))
# prediction=data.values[:, 10]
# keyword= data.values[:, 6]
# keyword=[cop.sub("", i) for i in keyword]
# print(seqeval_util.keyword_eval(keyword, prediction,'dic_removed'))
#
# print(seqeval_util.keyword_eval_part_v2(keyword, prediction,'dic_removed_partial'))
#
# data = pd.DataFrame(pd.read_excel('D:\Github\BERT-Keyword-Extractor\GKB_doc\en_finalresult_nodiction.xlsx', encoding='utf8'))
# prediction=data.values[:,10]
# keyword= data.values[:,6]
# #
# print(seqeval_util.keyword_eval(keyword, prediction,'orig_bert'))
# #
# print(seqeval_util.keyword_eval_part_v2(keyword, prediction,'orig_bert_partial'))
#
data = pd.DataFrame(pd.read_excel('D:/Github/BERT-Keyword-Extractor/GKB_doc/diction_bert.xlsx', encoding='utf8'))
prediction=data.values[:,10]
keyword= data.values[:,6]
#
print(seqeval_util.keyword_eval(keyword, prediction,'diction_bert'))
#
print(seqeval_util.keyword_eval_part_v2(keyword, prediction,'diction_bert_bert_partial'))