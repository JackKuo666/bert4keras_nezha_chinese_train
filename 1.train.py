#! -*- coding: utf-8 -*-
# 词级别的中文Nezha预训练
# MLM任务

import os

import json
import math
import random
import numpy as np
import tensorflow as tf
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras,注意，这里要放在引用keras之前
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_modelydfyhhbvvvv
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import text_segmentate
import jieba_fast as jieba



def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target



class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        for is_end, text in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            source, target = random_masking(token_ids)
            yield source, segment_ids, target


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy', aggregation='mean')
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


def corpus(data_path):
    """语料生成器 text_process
    """
    while True:
        with open(data_path,'r',encoding="utf-8") as f:
            for line in f:
                for l in text_segmentate(line, maxlen, seps=['\n', "。",","], strips=None):
                    yield l


def build_model(config_path, checkpoint_path):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        bert = build_transformer_model(
            config_path,
            checkpoint_path=None,
            model='nezha',
            with_mlm='linear',
            # ignore_invalid_weights=True,
            return_keras_model=False
        )
        model = bert.model

        # 训练用模型
        y_in = keras.layers.Input(shape=(None,), name='Input-Label')
        outputs = CrossEntropy(1)([y_in, model.output])

        train_model = keras.models.Model(model.inputs + [y_in], outputs)

        AdamW = extend_with_weight_decay(Adam, name='AdamW')
        AdamWLR = extend_with_piecewise_linear_lr(AdamW, name='AdamWLR')
        AdamWLRG = extend_with_gradient_accumulation(AdamWLR, name='AdamWLRG')
        optimizer = AdamWLRG(
            learning_rate=2e-5,
            weight_decay_rate=0.01,
            exclude_from_weight_decay=['Norm', 'bias'],
            grad_accum_steps=4,
            lr_schedule={20000: 1}
        )
        train_model.compile(optimizer=optimizer)
        train_model.summary()
        
        # 注意：从官网下载的bert模型需要以该方式加载
        bert.load_weights_from_checkpoint(checkpoint_path)  

        # tf.keras训练的模型ModelCheckpoint保存的需要以该方式加载
        # train_model.load_weights(checkpoint_path)


        return bert, train_model


class Evaluator(keras.callbacks.Callback):
    """训练回调
    """

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(model_save_checkpoint_path, overwrite=True)
        # bert.save_weights_as_checkpoint(model_save_checkpoint_path)  # 保存模型: 注意：这里有一些因为tf.keras升级带来的bug，暂时不使用这个


if __name__ == '__main__':

    # 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    # 获取所有 GPU 设备列表
    gpus = tf.config.experimental.list_physical_devices('GPU')


    if gpus:
        try:
            # 设置 GPU 显存占用为按需分配，增长式
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # 异常处理
            print(e)

    # bert配置 sss
    config_path = '/home/jack/code/bert_model/NEZHA_base_WWM/bert_config.json' 


    # 1.官网的预训练模型 需要bert.load_weights_from_checkpoint                    
    # 注意：这个只需要加载时使用 bert.load_weights_from_checkpoint，保存的模型都是需要用 train_model.load_weights 再次加载
    checkpoint_path = '/home/jack/code/bert_model/NEZHA_base_WWM/model.ckpt'  

    # 2.从Evaluator的self.model.save_weights 保存的模型：  需要train_model.load_weights
    # checkpoint_path = '../../data/bert_pretrain_patent/bert4keras_nezha/nezha_wwm_base_v2/model.ckpt.3/epoch_10_loss_0.39_acc_0.90.ckpt' # 前10w数据训练10epoch的结果

    # 3.从tf.keras.callbacks.ModelCheckpoint保存的模型  需要train_model.load_weights
    # checkpoint_path = '../../data/bert_pretrain_patent/bert4keras_nezha/nezha_wwm_base_v2/model.ckpt.2/variables/variables' 


    dict_path = '/home/jack/code/bert_model/NEZHA_base_WWM/vocab.txt'
    model_save_checkpoint_path = '../../data/bert_pretrain_patent/bert4keras_nezha/nezha_wwm_base_v2/model.ckpt.3/bert_model.ckpt' 
    # 关于保存地址：
    # 1.使用 tf.keras.callbacks.ModelCheckpoint 的时候： bert_model.ckpt 只表示地址，模型文件将会保存在该文件夹里面
    # 2.使用 Evaluator的self.model.save_weights 的时候：bert_model.ckpt 只表示文件名

    log_dir = '../../data/bert_pretrain_patent/bert4keras_nezha/nezha_wwm_base_v2/log_dir.2'


    # 基本训练参数
    maxlen = 512
    batch_size = 24
    epochs = 10
    total_line =  100000 # 1117452 

    # 数据处理
    # jieba.load_userdict("../../../cut_data/keyword.txt")
    jieba.load_userdict("data/keyword.txt")

    jieba.initialize()

    tokenizer = Tokenizer(dict_path, do_lower_case=True, pre_tokenize=lambda s: jieba.cut(s, HMM=False))

    data_path =  'data/100_line.csv'
    # data_path =  '../../data/bert_pretrain_patent/100_line.csv'
    # data_path =  '../../data/bert_pretrain_patent/10w_20w_line.csv'
    # data_path =  '../../data/bert_pretrain_patent/100w_line.csv'

    train_generator = data_generator(corpus(data_path), batch_size, 10**5)
    dataset = train_generator.to_dataset(
        types=('float32', 'float32', 'float32'),
        shapes=([None], [None], [None]),
        names=('Input-Token', 'Input-Segment', 'Input-Label'),
        padded_batch=True
    )


    # 启动训练
    bert, train_model = build_model(config_path, checkpoint_path)

    
    my_callbacks = [
            keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=2),
            keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.ModelCheckpoint(filepath=model_save_checkpoint_path+'/epoch_{epoch:02d}_loss_{loss:.2f}_acc_{accuracy:.2f}.ckpt',
                                                 save_weights_only=True,
                                                 monitor='loss',
                                                 mode='min',
                                                 save_best_only=True,
                                                 verbose=1),
            # Evaluator()
    ]
    step_per_epoch = total_line//batch_size+1
    print("step_per_epoch:", step_per_epoch)
    train_model.fit(
        dataset, epochs=epochs, steps_per_epoch=step_per_epoch, callbacks=my_callbacks
    ) # steps_per_epoch=1000


