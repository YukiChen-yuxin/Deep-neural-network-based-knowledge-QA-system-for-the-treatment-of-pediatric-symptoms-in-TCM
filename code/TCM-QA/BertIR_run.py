# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/train.txt'                                # 训练集
        self.dev_path = dataset + '/dev.txt'                                    # 验证集
        self.test_path = dataset + '/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/class_my.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/' + self.model_name + '_without_my.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        #self.num_epochs = 3                                             # epoch数
        self.num_epochs = 30
        #self.batch_size = 128
        self.batch_size = 8                                         # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 4e-6                                       # 学习率
        #self.bert_path = 'bert-base-chinese'
        self.bert_path = 'E:/毕业论文相关/毕业论文/代码/TCM-QA/bert-base-chinese/'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out



key = {
    0: 'med_use',
    1: 'med_con',
    2: 'med_sym',
    3: 'sym_med',
    4: 'unknown'
}


#config = Config('/content/drive/MyDrive/TCM-QA/IR data')
config = Config('E:\\毕业论文相关\\毕业论文\\代码\\TCM-QA\\IR data')
model = Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))


def build_predict_text(text):
    token = config.tokenizer.tokenize(text)
    token = ['[CLS]'] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    pad_size = config.pad_size
    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + ([0] * (pad_size - len(token)))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    ids = torch.LongTensor([token_ids]).cuda()
    seq_len = torch.LongTensor([seq_len]).cuda()
    mask = torch.LongTensor([mask]).cuda()
    return ids, seq_len, mask


def predict(text):
    """
    单个文本预测
    :param text:
    :return:
    """
    data = build_predict_text(text)
    with torch.no_grad():
        outputs = model(data)
        num = torch.argmax(outputs)
    return key[int(num)]
