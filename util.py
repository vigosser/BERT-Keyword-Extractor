import os
import logging
import csv
import pickle





logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class util(object):
    base_dir = 'D:\Github\BERT-Keyword-Extractor'
    csvpath = "result.txt"
    cache_dir='/workdir/pretrain-model/bert-torch'
    def set_csvpath(self, path):
        self.csvpath = path
        return self

    def save_csv_append(self, data):
        if not os.path.exists(os.path.dirname(self.csvpath)):
            os.makedirs(os.path.dirname(self.csvpath))
            logging.info("create folder:{}".format(os.path.dirname(self.csvpath)))
        with open(self.csvpath, 'a+', newline='', encoding='utf-8') as f:
            f.write(str(data)+"\r\n")
        return self

    def save_csv_ap(self, data, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            logging.info("create folder:{}".format(os.path.dirname(path)))
        with open(path, 'a+', newline='', encoding='utf-8') as f:
            f.write(str(data)+"\r\n")
        return self

    def save_txt(self, data, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            logging.info("create folder:{}".format(os.path.dirname(path)))
        with open(path, 'a+', newline='', encoding='utf-8') as f:
            f.write(str(data)+"\r\n")
        return self

    def to_sentence(self, sentences,path):
        sent = ''
        for i in sentences:
            sent = sent + i + ' '
        utiler.save_txt(sent, path)

    def to_tags(self, labels, path):
        sent = ''
        for i in labels:
            sent = sent + i + ' '
        utiler.save_txt(sent, path)

    def save_variable(self, v, filename):
        f = open(filename, 'wb')
        pickle.dump(v, f)
        f.close()
        return filename

    def load_variavle(self, filename):
        f = open(filename, 'rb')
        r = pickle.load(f)
        f.close()
        return r


utiler = util()
