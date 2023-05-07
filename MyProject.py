
import os
import re
import numpy as np
from keras.utils import np_utils
from keras.layers import LSTM,Dense,TimeDistributed,Bidirectional
from keras.models import Sequential,Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import *
from keras import initializers,constraints,activations,regularizers
import keras.backend as K
from keras.utils import np_utils
import tensorflow.compat.v1 as tf
tf.disable_eager_execution() #禁用Tensorflow2 默认的即时执行模式
#from keras.utils import pad_sequences
#keras2.4.3版本引入pad_sequences方法
from keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')
from transformers import BertTokenizer,TFBertModel,BertModel
from keras import optimizers
import tensorflow as tf
import json
#https://blog.csdn.net/yjw123456/article/details/120232707
class LayerPlus(Layer):
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            })
        return config

    def reuse(self,layer,*args,**kwargs):
        if not layer.built:
            if len(args) >0:
                inputs=args[0]
            else:
                inputs=kwargs['inputs']
            if isinstance(inputs,list):#获取输入的形状
                input_shape=[K.int_shape(x) for x in inputs]
            else:
                input_shape=K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args,**kwargs)

        for w in layer.trainable_weights:
            exist=False
            for _w in self.non_trainable_weights:
                if tf.equal(w,_w):
                    exist=True
            if exist ==False:
                self._trainable_weights.append(w)
            #if w not in self._trainable_weights:
            #    self._trainable_weights.append(w)

        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)

        for u in layer.updates:
            if not hasattr(self,'_updates'):
                self._updates=[]
            if u not in self._updates:
                self._updates.append(u)
        return outputs

class Attention(LayerPlus):
    def __init__(self, out_dim,key_size=8, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.key_size = key_size
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'out_dim': self.out_dim,
            'key_size': self.key_size,
        })
        return config

    def build(self,input_shape):
        super(Attention,self).build(input_shape)
        input_shape = list(input_shape)
        if input_shape[1] == None:
            input_shape[1] = 1
        kernel_initializer = 'glorot_uniform'
        kernel_regularizer = None
        kernel_constraint = None
        self.query_weight = self.add_weight(name='qw',
                                  shape=(input_shape[1], self.out_dim),
                                  # shape=(1, self.out_dim),
                                  initializer=kernel_initializer,
                                  regularizer=kernel_regularizer,
                                  constraint=kernel_constraint,
                                  trainable=True)
        self.key_weight = self.add_weight(name='kw',
                                  shape=(input_shape[1], self.out_dim),
                                  # shape=(1, self.out_dim),
                                  initializer=kernel_initializer,
                                  regularizer=kernel_regularizer,
                                  constraint=kernel_constraint,
                                  trainable=True)
        self.value_weight = self.add_weight(name='vw',
                                  shape=(input_shape[1], self.out_dim),
                                  # shape=(1, self.out_dim),
                                  initializer=kernel_initializer,
                                  regularizer=kernel_regularizer,
                                  constraint=kernel_constraint,
                                  trainable=True)
        self.built = True

    def call(self,inputs):
        #query,key,value形状为m x n,key的转置形状为n x m，query与key的转置相乘形状为 m x m
        input_size = tf.shape(inputs)
        query = tf.multiply(inputs,self.query_weight) #query为input与query权重相乘
        key = tf.multiply(inputs,self.key_weight) #key为input与key权重相乘
        value = tf.multiply(inputs,self.value_weight) #value为input与value权重相乘
        key = K.permute_dimensions(key,(0,2,1)) #将key的矩阵转置。也就是第2维和第1维交换
        p = tf.matmul(query,key) #将quert与key的转置点乘，得到中间结果p
        p = p/np.sqrt(self.key_size) #将点乘结果进行缩放
        p = K.softmax(p) #将缩放结果经过softmax处理
        p = tf.reshape(p,(input_size[0],input_size[1],input_size[1],1)) #改变形状
        value = tf.tile(value,[1,input_size[1],1])#将value的第1维复制
        value = tf.reshape( value , (input_size[0],input_size[1],input_size[1],self.out_dim) )
        p = tf.multiply(value,p)
        v = tf.reduce_sum(p,2) #将第2维上所有元素求和合并，3维将第0维求和，相当于将所有矩阵求和，第1维求和相当于将每个矩阵所有行合并，第2维求和相当于将每个矩阵所有列合并。
        return v

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.out_dim)

class MultiAttention(LayerPlus):
    def __init__(self,out_dim,heads,**kwargs):
        super(MultiAttention,self).__init__(**kwargs)
        self.out_dim=out_dim
        self.heads=heads
        self.attentionHead=[]
        self.built = False

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'out_dim': self.out_dim,
            'heads': self.heads,
            'attentionHead': self.attentionHead,
        })
        return config

    def build(self,input_shape,):
        super(MultiAttention,self).build(input_shape)
        for i in range(0,self.heads):
            self.attentionHead=self.attentionHead+[Attention(out_dim=self.out_dim)]
        self.w0=Dense(units=self.out_dim,use_bias=False) #Dense输出为output维度
        self.built = True

    def call(self,inputs):
        for i in range(self.heads):
            self.attentionHead[i]=self.reuse(self.attentionHead[i],inputs)

        h=tf.concat(self.attentionHead,-1)
        h=self.reuse(self.w0,h)
        return h

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.out_dim)

def ConverToNumpy(tensor):
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        cls=sess.run(tensor)
    return cls
def PreProcessWithBert(data):
    bert_model = TFBertModel.from_pretrained('../bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')
    bertOutput1 = None
    bertOutput2 = None
    token = tokenizer(data[:7000],return_tensors='tf',max_length=50,padding='max_length',truncation=True,add_special_tokens=True)
    output = bert_model(token)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        bertOutput1 = sess.run(output['last_hidden_state'])
    print(bertOutput1.shape)
    token = tokenizer(data[7000:], return_tensors='tf', max_length=50, padding='max_length', truncation=True,
                      add_special_tokens=True)
    output = bert_model(token)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        bertOutput2 =sess.run(output['last_hidden_state'])
    print(bertOutput2.shape)
    return np.concatenate((bertOutput1,bertOutput2),axis=0)

def createModel(train_data):
    inputLayer = Input(shape=(train_data.shape[1], train_data.shape[2]))
    batchNormalization = BatchNormalization()(inputLayer)
    dense1 = Dense(units=64, activation='relu')(batchNormalization)
    dropout1 = Dropout(0.3)(dense1)
    pool1 = MaxPooling1D(pool_size=5,strides=None,padding='valid')(dropout1)
    dense2 = Dense(units=128,activation='relu')(pool1)
    dropout2 = Dropout(0.3)(dense2)
    fla = Flatten()(dropout2)
    OutputLayer = Dense(units=3, activation='softmax')(fla)
    model = Model(inputs=inputLayer, outputs=OutputLayer)
    sgd = optimizers.gradient_descent_v2.SGD(learning_rate=0.001)
    #adam = optimizers.adam_v2.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model

import matplotlib.pyplot as plt
def show_train_history(history,train,validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('Train_History')
    plt.xlabel('Epoch')
    plt.ylabel(train)
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

def TrainModel(model,train_data,train_label,test_data,test_label):
    histoty=model.fit(
        x=train_data,
        y=train_label,
        batch_size=32,
        epochs=100,
        validation_split=0.2,
        verbose=1
    )
    model.save_weights('BertRestNet503.h5')
    score= model.evaluate(x=test_data,y=test_label)
    print(score[1])
    with open('BertRestNet50SGD3.json','w') as f:
        json.dump(histoty.history,f)
   # show_train_history(histoty, 'loss', 'val_loss')

def TestModel(model,test_data,test_label):
    score = model.evaluate(x=test_data,y=test_label)
    print(score[1])

def CreateBiLSTMModel(train_data):
    inputLayer = Input(shape=(train_data.shape[1], train_data.shape[2]))
    batchNormalization = BatchNormalization()(inputLayer)
    BiLSTM = Bidirectional(LSTM(train_data.shape[1], return_sequences=True), merge_mode='concat',
                           input_shape=(train_data.shape[1], train_data.shape[2]))(batchNormalization)
    AttentionLayer = MultiAttention(out_dim=train_data.shape[1]*2, heads=8)(BiLSTM)
    fla = Flatten()(AttentionLayer)
    OutputLayer = Dense(units=3, activation='softmax')(fla)
    model = Model(inputs=inputLayer, outputs=OutputLayer)
    sgd = optimizers.gradient_descent_v2.SGD(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model

def TrainBiLSTMModel(model,train_data,train_label,test_data,test_label):
    histoty = model.fit(
        x=train_data,
        y=train_label,
        batch_size=64,
        epochs=20,
        validation_split=0.2,
        verbose=1
    )
    model.save_weights('BiLSTM.h5')
    score=model.evaluate(x=test_data,y=test_label)
    print(score[1])



#https://zhuanlan.zhihu.com/p/107737824


from transformers import BeitImageProcessor,BeitForImageClassification
from PIL import Image
from keras.applications.resnet import ResNet50
from keras.applications.resnet import  preprocess_input
def readfile(path):
    label_map = {'positive': 0,
                 'negative': 1,
                 'neutral': 2
                 }
    labelPath=path+"/labelResultAll.txt"
    dataPath = path + "/data"
    lineindex=0; #跳过第一行
    text_data=[]
    image_path=[]
    label=[]
    canRead=False
    with open(labelPath,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if lineindex == 0:
                lineindex = lineindex + 1;
                continue;
            content=line.replace('\n','')
            content=content.split("\t")
            canRead=False
            if content[1]==content[2] or content[1]==content[3] or content[2]==content[3]:#三对标签至少有两对才能去
                if content[1]==content[2]:
                    l = content[1].split(',') #按照论文中做法，图文标签不一致则舍弃掉
                    if l[0]==l[1]: #只取图文标签一致的数据
                        canRead = True #表明该条数据可以取
                        label=label+[label_map[l[0]]]
                elif content[1]==content[3]:
                    l = content[1].split(',')  # 按照论文中做法，图文标签不一致则舍弃掉
                    if l[0] == l[1]:
                        canRead = True
                        label = label + [label_map[l[0]] ]

                elif content[2]==content[3]:
                    l = content[2].split(',')  # 按照论文中做法，图文标签不一致则舍弃掉
                    if l[0] == l[1]:
                        canRead = True
                        label = label + [ label_map[l[0]] ]

                if canRead==True:
                    with open(dataPath + '/' + content[0] + '.txt', 'r', encoding='utf-8') as t:
                        line = t.readline()
                        re_tag = re.compile(r'(#\S+)|(\s{2,})|(@)')
                        re_tag.sub('',line)
                        text_data = text_data + [line]
                    image_path = image_path + [dataPath + '/' + content[0] + '.jpg']

    return text_data,image_path,label

def extractFromBiLSTM(train_data):
    BiLSTM = CreateBiLSTMModel(train_data)
    BiLSTM.load_weights('BiLSTM.h5')
    model = Model(inputs=BiLSTM.input, outputs=BiLSTM.get_layer('multi_attention').output)
    feature=model.predict(train_data)
    return np.array(feature)

def PreprocessImage(pathLit):
    res50 = ResNet50(weights='imagenet',   include_top=True, input_shape=(224, 224, 3))
    model = Model(inputs=res50.input, outputs=res50.get_layer('avg_pool').output)
    picFeatureList=[]
    for Item in pathLit:
        picFeatureList=picFeatureList+[PicturePretrained(model, Item)]
    return np.array(picFeatureList)

def PicturePretrained(model,path):
    img = Image.open(path)
    img = img.resize((224, 224))
    x = np.array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features=np.resize(features,(50,41))#原本尺寸为2048=2048,改变形状为50X41=2050
    return features
def BertBiLSTMRun():
    label = np.load('label.npy')
    image_pretrained=np.load('image_pretrained.npy')
    text_data_extract_BiLSTM=np.load('text_data_extract_BiLSTM.npy')
    concated_data=np.concatenate(( text_data_extract_BiLSTM ,image_pretrained),axis=2)
    index = [i for i in range(len(concated_data))]
    concated_data=concated_data[index]
    label = label[index]
    label_Onehot = np_utils.to_categorical(label)
    train_data=concated_data[:8000]
    test_data=concated_data[8000:]
    train_label =label_Onehot[:8000]
    test_label = label_Onehot[8000:]
    print("train data shape: ",train_data.shape)
    print("train label shape: ", train_label.shape)
    print("test data shape: ", test_data.shape)
    print("test label shape: ", test_label.shape)
    model=createModel(train_data)
    #plot_model(model,'ResNet50_Bert_Multiattention.png',show_shapes=True)
    #model.load_weights('BertRestNet502.h5')
    #score = model.evaluate(x=test_data,y=test_label)
    #print(score[1])
    TrainModel(model,train_data,train_label,test_data,test_label)

BertBiLSTMRun()
#print(tf.test.is_gpu_available())
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
