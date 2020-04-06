import os
import logging
import csv
import progressbar
from util import utiler


class seqeval_util(object):
    # 要求準確命中
    def keyword_eval(self, keywords, predictions, save_path):
        print(save_path)
        ace = 0
        pre_d = 0
        recall_d = 0
        with progressbar.ProgressBar(max_value=100) as bar:
            for j, keyword in enumerate(keywords):
                keyword = keyword.replace('keywords:', '')
                bar.update(round(j / len(keywords), 6) * 100)
                if predictions[j] != predictions[j]:
                    predictions[j] = ''
                    continue
                predictions[j] = predictions[j].replace(' - ', '-')
                keys = keyword.split(';')
                recall_d = recall_d + len(keys)
                pre_d = pre_d + len(predictions[j].split(';'))
                temmmp = 0
                # 把与测结果分词，取出空格
                prediction_list = []
                for p in predictions[j].split(';'):
                    if p != '':
                        prediction_list.append(p.strip())
                for i in keys:
                    # 关键字存在
                    if i.strip() in prediction_list:
                        ace = ace + 1
                        temmmp = temmmp + 1
                utiler.save_csv_ap(
                    keyword + '|' + predictions[j] + '|' + 'recalled:' + str(temmmp / len(keys)) + '|pre:' + \
                    str(temmmp / len(prediction_list)), './result/{}.csv'.format(save_path))
                # print('keyword:' + keyword)
                # print('prediction:' + predictions[j])
                # print('pre:' + temmmp / len(keys))
                # print('recall:' + temmmp / len(predictions[j].split(';')))
            print(pre_d)
            print(recall_d)
        return ace / recall_d, ace / pre_d, self.f1(ace / recall_d, ace / pre_d)

    def f1(self, pre, recall):
        assert pre <= 1
        assert recall <= 1
        return 2 * (pre * recall) / (pre + recall)

    # 部分命中也算對

    def keyword_eval_part_v2(self, keywords, predictions, save_path):
        print(save_path)
        ace = 0
        pre_d = 0
        recall_d = 0
        pre_acc = 0
        with progressbar.ProgressBar(max_value=100) as bar:
            for j, keyword in enumerate(keywords):
                bar.update(round(j / len(keywords), 6) * 100)
                keyword = keyword.replace('keywords:', '')
                if predictions[j] != predictions[j]:
                    predictions[j] = ''
                    continue
                predictions[j] = predictions[j].replace(' - ', '-')
                keys = keyword.strip().split(';')
                key_word_list = keyword.strip().replace(';', ' ').split(' ')
                recall_d = recall_d + len(keys)
                pre_d = pre_d + len(predictions[j].split(';'))
                tmp_acc = 0
                tmp_pre_acc = 0

                # 把预测结果分词，去掉空格

                for key in keys:
                    # 关键字存在
                    if key == '':
                        continue
                    for word in key.strip().split(' '):
                        if word.strip() in predictions[j]:
                            ace = ace + 1
                            tmp_acc = tmp_acc + 1
                            break
                prediction_list = []
                for p in predictions[j].split(';'):
                    if p != '':
                        prediction_list.append(p.strip())
                for prediction in prediction_list:
                    # 关键字存在
                    if prediction == '':
                        continue
                    for word in prediction.strip().split(' '):
                        if word.strip() in key_word_list:
                            pre_acc = pre_acc + 1
                            tmp_pre_acc = tmp_pre_acc + 1
                            break
                utiler.save_csv_ap(
                    keyword + '|' + predictions[j] + '|' + 'recall:' + str(tmp_acc / len(keys)) + '|pre:' + \
                    str(tmp_pre_acc / len(predictions[j].split(';'))), './result/{}.csv'.format(save_path))
                # print('keyword:' + keyword)
                # print('prediction:' + predictions[j])
                # print('pre:' + temmmp / len(keys))
                # print('recall:' + temmmp / len(predictions[j].split(';')))
            print(pre_d)
            print(recall_d)
        return ace / recall_d, pre_acc / pre_d, self.f1(ace / recall_d, ace / pre_d)


seqeval_util = seqeval_util()
