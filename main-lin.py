# coding: utf-8

# # Keyword-Extraction using BERT

# Use BERT Token Classification Model to extract keyword tokens from a sentence.

# ## Prepare Dataset for BERT.
#
# Convert key-text recognition dataset to BIO format dataset.


import os
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
from util import utiler
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
import re
import progressbar
import pickle
from seqeval.metrics import f1_score, accuracy_score, precision_score, recall_score

tag2idx = {'B': 0, 'I': 1, 'O': 2}
tags_vals = ['B', 'I', 'O']
# nltk.download('punkt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

parser = argparse.ArgumentParser(description='BERT Keyword Extraction Model')
parser.add_argument('--data', type=str, default='/workdir/data/GRB_training_text_en_all.xlsx',
                    help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=4,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=512, metavar='N',
                    help='sequence length')
parser.add_argument('--lr', type=float, default=3e-5,
                    help='initial learning rate')
parser.add_argument('--save', type=str, default='/workdir/data/en_model.pt',
                    help='path to save the final model')
parser.add_argument('--lang', type=str, default='en',
                    help='path to save the final model')
parser.add_argument('--result_path', type=str, default='/workdir/result.txt',
                    help='path to the result')
parser.add_argument('--key_path', type=str, default='/workdir/githc/b-k-e/GKB_doc/tag_count_all.csv',
                    help='path to the key_dictionary')
parser.add_argument('--temp_path', type=str, default='/workdir/temp',
                    help='path to the key_dictionary')

args = parser.parse_args()
MAX_LEN = args.seq_len
bs = args.batch_size
utiler.set_csvpath(args.result_path)
if args.lang == "en":
    bert_model = "bert-base-uncased"
elif args.lang == "cn":
    bert_model = "bert-base-chinese"
elif args.lang == "mutil":
    bert_model = "bert-base-multilingual-uncased"
bert_model = 'bert-base-uncased'
args.data = 'D:/Github/BERT-Keyword-Extractor/GKB_doc/GRB_training_text_en_all.xlsx'
args.key_path = 'D:/Github/BERT-Keyword-Extractor/GKB_doc/tag_count_all.csv'
args.temp_path='D:/Github/BERT-Keyword-Extractor'
args.usedict=True
data = pd.DataFrame(pd.read_excel(args.data))
key_d = pd.DataFrame(pd.read_csv(args.key_path)).values[:, 0]


def flat_accuracy(preds, labels):
    utiler.save_csv_append(str(preds.shape))
    utiler.save_csv_append("------------preds-------------")
    utiler.save_csv_append(preds)
    pred_flat = np.argmax(preds, axis=2).flatten()
    utiler.save_csv_append(str(pred_flat.shape))
    utiler.save_csv_append("------------preds-faltter-------------")
    utiler.save_csv_append(pred_flat)
    labels_flat = labels.flatten()
    utiler.save_csv_append(str(labels.shape))
    utiler.save_csv_append("------------labels-------------")
    utiler.save_csv_append(labels)
    utiler.save_csv_append(str(labels_flat.shape))
    utiler.save_csv_append("------------labels_flat-------------")
    utiler.save_csv_append(labels_flat)
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def extend_single(labels, sentences, tokenizer):
    words = []
    for i in range(len(labels)):
        sent = word_tokenize(sentences[i])
        moves = 0
        for j, word in enumerate(sent):
            tk_words = tokenizer.tokenize(word)
            words = words + tk_words
            if len(tk_words) != 1:
                for add in range(len(tk_words) - 1):
                    labels[i].insert(j + 1 + moves, 'I' if labels[i][j + moves] != 'O' else 'O')
            moves = moves + len(tk_words) - 1

    return labels, words


def convert_with_diction(sentence):
    token = sentence
    key_sent = []
    labels = []
    # for token in tokens:
    sent = word_tokenize(token.lower())
    z = ['O'] * len(sent)
    for k in key_d:
        if k in token:
            k = k.lower()
            if len(k.split()) == 1:
                try:
                    z[sent.index(k.split()[0])] = 'B'
                except ValueError:
                    continue
            elif len(k.split()) > 1:
                try:
                    if sent.index(k.split()[0]) and sent.index(
                            k.split()[-1]):
                        z[sent.index(k.split()[0])] = 'B'
                        for j in range(1, len(k.split())):
                            z[sent.index(k.split()[j])] = 'I'
                except ValueError:
                    continue
    for m, n in enumerate(z):
        if z[m] == 'I' and z[m - 1] == 'O':
            z[m] = 'O'
    labels.append(z)
    key_sent.append(token)
    return key_sent, labels


def convert(sentence, key):
    # tokens = sent_tokenize(sentence)
    token = sentence
    keys = str(key).split(';')
    key_sent = []
    labels = []
    # for token in tokens:
    sent = word_tokenize(token.lower())
    z = ['O'] * len(sent)
    for k in keys:
        if k in token:
            k = k.lower()
            if len(k.split()) == 1:
                try:
                    z[sent.index(k.split()[0])] = 'B'
                except ValueError:
                    continue
            elif len(k.split()) > 1:
                try:
                    if sent.index(k.split()[0]) and sent.index(
                            k.split()[-1]):
                        z[sent.index(k.split()[0])] = 'B'
                        for j in range(1, len(k.split())):
                            z[sent.index(k.split()[j])] = 'I'
                except ValueError:
                    continue
    for m, n in enumerate(z):
        if z[m] == 'I' and z[m - 1] == 'O':
            z[m] = 'O'
    labels.append(z)
    key_sent.append(token)
    return key_sent, labels


sentences_all = []
labels_all = []
input_ids_all = []
tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True, cache_dir=utiler.cache_dir)

if  os.path.exists(args.temp_path+'/sentence_all.pkl'):
    with progressbar.ProgressBar(max_value=100) as bar:
        for i in range(len(data)):
            sentences_ = []
            labels_ = []
            input_ids = []
            bar.update(i / len(data) * 100)
            cop = re.compile("[^ !#$%&\'()*+,-./:;<=>?\\^_`{|}~^a-z^A-Z^0-9]")
            html_sub = re.compile('<\\/?.+?\\/?>')  # html标签
            if data.values[i, 8] != data.values[i, 8]:
                continue
            en_text = data.values[i, 8].replace('``', '').replace("\"", '').replace('\uf06c', '')
            tokens = sent_tokenize(en_text)
            for iii, text in enumerate(tokens):
                text = cop.sub("", text)  ## replace unnecessary char.
                text = html_sub.sub("", text)
                if args.usedict:
                    s, l = convert_with_diction(text)
                else:
                    s,l = convert(text,)
                l, tks = extend_single(l, s, tokenizer)
                ids = tokenizer.convert_tokens_to_ids(tks)
                assert len(tks) == len(l[0])
                assert len(ids) == len(l[0])
                sentences_ = sentences_ + tks
                labels_ = labels_ + l[0]
                input_ids = input_ids + ids
            if not 'B' in labels_ and not 'I' in labels_:
                continue
            sentences_all.append(sentences_)
            labels_all.append(labels_)
            input_ids_all.append(input_ids)
    utiler.save_variable(sentences_all, args.temp_path+'/sentence_all.pkl')
    utiler.save_variable(labels_all, args.temp_path+'/labels_all.pkl')
    utiler.save_variable(input_ids_all, args.temp_path+'/input_ids_all.pkl')
    df = pd.DataFrame(columns=['sentence', 'labels'])
    df['sentence'] = sentences_all
    df['labels'] = labels_all
    df.to_excel('temp_all.xlsx')
else:
    sentences_all = utiler.load_variavle(args.temp_path+'/sentence_all.pkl')
    labels_all = utiler.load_variavle(args.temp_path+'/labels_all.pkl')
    input_ids_all = utiler.load_variavle(args.temp_path+'/input_ids_all.pkl')
input_ids = pad_sequences(input_ids_all, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels_all], maxlen=MAX_LEN, value=tag2idx["O"],
                     padding="post", dtype="long", truncating="post")

attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

model = BertForTokenClassification.from_pretrained(bert_model, num_labels=len(tag2idx),cache_dir=utiler.cache_dir)

model = model.cuda()
# model = torch.nn.DataParallel(model)

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=args.lr)

epochs = args.epochs
max_grad_norm = 1.0

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss / nb_tr_steps))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

torch.save(model, args.save)

model.eval()
predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None,
                       attention_mask=b_input_mask)

    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

    label_ids = b_labels.to('cpu').numpy()
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy
    utiler.set_csvpath(args.result_path).save_csv_append("flat_accuracy:{}".format(flat_accuracy(logits, label_ids))). \
        save_csv_append('_accuracy_score:{}'.format(accuracy_score(pred_tags, valid_tags))). \
        save_csv_append('_precision_score:{}'.format(precision_score(pred_tags, valid_tags))). \
        save_csv_append('_recall_score:{}'.format(recall_score(pred_tags, valid_tags)))
    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l]
utiler.save_csv_append("Validation loss: {}".format(eval_loss / nb_eval_steps))
utiler.save_csv_append("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
utiler.save_csv_append("-----------pred_tags------------")
utiler.save_csv_append(pred_tags)
utiler.save_csv_append("-----------valid_tags------------")
utiler.save_csv_append(valid_tags)
utiler.save_csv_append("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
