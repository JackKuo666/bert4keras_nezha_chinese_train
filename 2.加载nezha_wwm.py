# 设置TF_KERAS = 1，表示使用tf.keras
import os
os.environ["TF_KERAS"] = '1'

import numpy as np
from tensorflow.keras.models import load_model
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array

# 环境设置
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 模型保存路径
checkpoint_path = "/home/jack/code/bert_model/chinese_L-12_H-768_A-12/bert_model.ckpt"
dict_path = "/home/jack/code/bert_model/chinese_L-12_H-768_A-12/vocab.txt"
config_path = "/home/jack/code/bert_model/chinese_L-12_H-768_A-12/bert_config.json"

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='bert',
    return_keras_model=True
)

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')
token_ids, segment_ids = to_array([token_ids], [segment_ids])

# 模型预测
print('\n ===== predicting =====\n')
print(bert.predict([token_ids, segment_ids]))

# 模型保存与加载
print('\n ===== reloading and predicting =====\n')
bert.save('save.model')

del bert
bert = load_model('save.model')
print(bert.predict([token_ids, segment_ids]))

# https://blog.csdn.net/qq_41982466/article/details/123222564