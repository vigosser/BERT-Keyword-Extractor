from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
import torch
import argparse
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='BERT Keyword Extractor')
parser.add_argument('--sentence', type=str, default=' ',
                    help='sentence to get keywords')
parser.add_argument('--path', type=str, default='model.pt',
                    help='path to load model')
parser.add_argument('--lang', type=str, default='en',
                    help='path to save the final model')
args = parser.parse_args()
if args.lang == "en":
    bert_model = "bert-base-uncased"
    model_path = "/workdir/pretrain-model/bert-torch"
elif args.lang == "cn":
    bert_model = "bert-base-chinese"
    model_path = "/workdir/pretrain-model/bert-torch-cn"
bert_model = "bert-base-uncased"
model_path = "D:/Github/BERT-Keyword-Extractor/model/en_model.pt"
tag2idx = {'B': 0, 'I': 1, 'O': 2}
tags_vals = ['B', 'I', 'O']

tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
# cache_dir='/workdir/pretrain-model/bert-torch-cn')
model = BertForTokenClassification.from_pretrained(bert_model, num_labels=len(tag2idx))


# cache_dir='/workdir/pretrain-model/bert-torch-cn')
def casting(tkns, prediction):
    # 把命中peace的值扩张到整个单词
    for k, j in enumerate(prediction):
        if j == 1 or j == 0:
            # 中间是短词
            if not tkns[k].find('##') == -1:
                prediction[k] = 1
                forwd = False
                backward = False
                for i in range(int(len(tkns) / 2)):
                    # forward
                    if not tkns[k - i].find('##') == -1:
                        prediction[k - i] = 1
                    else:
                        prediction[k - i] = 0
                        forwd = True
                    # backward
                    if tkns[k + i].find('##') == -1:
                        prediction[k + i] = 1
                    else:
                        backward = True
                    if forwd & backward:
                        break
    peases = []
    begin = False
    for k, j in enumerate(prediction):
        if j == 0 or j == 1:
            begin = True
            # 有#的话是拼接
            if tkns[k].find('##') != -1:
                peases[len(peases) - 1] = peases[len(peases) - 1] + tkns[k].replace('#', '')
            else:
                peases.append(tkns[k])
        if j == 2 and prediction[k - 1] == 1 and begin==True:
            begin = False
            peases.append(';')
    return ' '.join(peases)


def keywordextract(sentence, model):
    # load model

    model.eval()
    prediction = []

    text = sentence
    # tkns=sent_tokenize(text)
    # tkns= tokenizer.basic_tokenizer.tokenize(text)
    tkns = tokenizer.tokenize(text)
    print(tkns)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tkns)
    segments_ids = [0] * len(tkns)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)

    logit = model(tokens_tensor, token_type_ids=None,
                  attention_mask=segments_tensors)
    logit = logit.detach().cpu().numpy()
    prediction.extend([list(p) for p in np.argmax(logit, axis=2)])
    result = ""
    flag = False
    peaces = []
    assert len(prediction[0]) == len(tkns)
    result = casting(tkns, prediction[0])

    # for k, j in enumerate(prediction[0]):
    #     peaces.append(tokenizer.convert_ids_to_tokens(tokens_tensor[0].to('cpu').numpy())[k])
    #     if j == 1 or j == 0:
    #
    #         flag = True
    #         # print(tokenizer.convert_ids_to_tokens(tokens_tensor[0].to('cpu').numpy())[k], j)
    #         if (tokenizer.convert_ids_to_tokens(tokens_tensor[0].to('cpu').numpy())[k].find('#') is not -1):
    #             sdfasd = 1
    #         result = result + tokenizer.convert_ids_to_tokens(tokens_tensor[0].to('cpu').numpy())[k] + " "
    #     elif flag == True:
    #         result = result + "; "
    #         flag = False
    print(result)
    return result, prediction


data = pd.DataFrame(pd.read_excel('D:/Github/BERT-Keyword-Extractor/GKB_doc/GRB_sample_clear.xlsx', encoding='utf8'))
en_text = data.values[:, 6]
en_tag = data.values[:, 5]
data['result_en'] = ""
data['result_en_prediction'] = ""
args.path = 'D:/Github/BERT-Keyword-Extractor/model/en_model.pt'
model = torch.load(args.path, map_location=device)
for i in range(0, len(en_text)):
    data['result_en'][i]
    input = ""

    if len(en_text[i]) >= 512:
        input = en_text[i][0:512]
    else:
        input = en_text[i]
    sentens = sent_tokenize(input)
    res, predictions = keywordextract(input, model)
    data['result_en'][i] = res
    data['result_en_prediction'][i] = predictions

data.to_excel("D:/Github、BERT-Keyword-Extractor/GKB_doc/cn_finalresult.xlsx")
