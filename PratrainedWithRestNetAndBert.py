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
from keras.callbacks import EarlyStopping #提前停止训练
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
    bert_model = TFBertModel.from_pretrained('./bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    bertOutput1 = None
    bertOutput2 = None
    token = tokenizer(data[:7000],return_tensors='tf',max_length=50,padding='max_length',truncation=True,add_special_tokens=True)

    output = bert_model(token)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        bertOutput1 = sess.run(output['last_hidden_state'])
    token = tokenizer(data[7000:], return_tensors='tf', max_length=50, padding='max_length', truncation=True,
                      add_special_tokens=True)

    output = bert_model(token)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        bertOutput2 =sess.run(output['last_hidden_state'])

    pretrained_data = np.concatenate((bertOutput1,bertOutput2),axis=0)
    np.save('text_data_pretrained.npy',pretrained_data)
    return output

def createModel(train_data):
    
    '''
    inputLayer = Input(shape=(train_data.shape[1], train_data.shape[2]))
    batchNormalization = BatchNormalization()(inputLayer)
    dense1 = Dense(units=64, activation='relu')(batchNormalization)
    dropout1 = Dropout(0.3)(dense1)
    pool1 = AveragePooling1D(pool_size=5,strides=None,padding='valid')(dropout1)
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
    '''
    inputLayer = Input(shape=(train_data.shape[1], train_data.shape[2]))
    batchNormalization = BatchNormalization()(inputLayer)
    dense1 = Dense(units=64, activation='relu')(batchNormalization)
    dropout1 = Dropout(0.3)(dense1)
    dense2 = Dense(units=128,activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense2)
    pool1 = AveragePooling1D(pool_size=5,strides=None,padding='valid')(dropout2)
    fla = Flatten()(pool1)
    OutputLayer = Dense(units=3, activation='softmax')(fla)
    model = Model(inputs=inputLayer, outputs=OutputLayer)
    sgd = optimizers.gradient_descent_v2.SGD(learning_rate=0.01)
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

#最好的效果参数epochs=200,

def TrainModel(model,train_data,train_label,test_data,test_label):
    earlyStop = EarlyStopping(monitor='val_accuracy',min_delta=0,patience=100,mode='max',verbose=1,restore_best_weights=True)
    histoty=model.fit(
        x=train_data,
        y=train_label,
        batch_size=120,
        callbacks=[earlyStop],
        epochs=500,
        validation_split=0.2,
        verbose=1,
        shuffle=True
    )
    model.save_weights('BertRestNet50FineTune.h5')
    score= model.evaluate(x=test_data,y=test_label)
    print(score[1])
    with open('BertRestNet50FineTune.json','w') as f:
        json.dump(histoty.history,f)

def TestModel(model,test_data,test_label):
    score = model.evaluate(x=test_data,y=test_label)
    print(score[1])

def CreateBiLSTMModel(train_data):
    inputLayer = Input(shape=(train_data.shape[1], train_data.shape[2]))
    batchNormalization = BatchNormalization()(inputLayer)
    BiLSTM = Bidirectional(LSTM(train_data.shape[1], return_sequences=True), merge_mode='concat',
                           input_shape=(train_data.shape[1], train_data.shape[2]))(batchNormalization)
    AttentionLayer = MultiAttention(out_dim=train_data.shape[1]*2, heads=8)(BiLSTM)
    dropout1 = Dropout(0.3)(AttentionLayer)
    fla = Flatten()(dropout1)
    OutputLayer = Dense(units=3, activation='softmax')(fla)
    model = Model(inputs=inputLayer, outputs=OutputLayer)
    sgd = optimizers.gradient_descent_v2.SGD(learning_rate=0.01)
    #adam = optimizers.adam_v2.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model

def TrainBiLSTMModel(model,train_data,train_label,test_data,test_label):
    earlyStop = EarlyStopping(monitor='val_accuracy',min_delta=0,patience=100,mode='max',verbose=1,restore_best_weights=True)
    histoty = model.fit(
        x=train_data,
        y=train_label,
        callbacks=[earlyStop],
        batch_size=80,
        epochs=500,
        validation_split=0.3,
        verbose=1,
        shuffle=True
    )
    model.save_weights('BiLSTM.h5')
    score = model.evaluate(x=test_data,y=test_label)
    print("BiLSTM accuracy:",score[1])
    return model



#https://zhuanlan.zhihu.com/p/107737824


from transformers import BeitImageProcessor,BeitForImageClassification
from PIL import Image
from keras.applications.resnet import ResNet50
from keras.applications.vgg19 import VGG19
import keras.applications.vgg19 as vgg19
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

def RestNetFineTune(image_path,label):
    imagelist=[]
    label_Onehot = []
    if os.path.exists('image_input.npy') and os.path.exists('label_finetune_restnet.npy'):
        imagelist= np.load('image_input.npy')
        label_Onehot = np.load('label_finetune_restnet.npy')
    else:
        np.random.seed(999)
        np.random.shuffle(image_path)
        np.random.seed(999)
        np.random.shuffle(label)
        for path in image_path:
            img = Image.open(path)
            img = img.resize((224,224),Image.ANTIALIAS)
            x = np.array(img)
            #print(x.shape)
            imagelist = imagelist+[preprocess_input(x)]
        imagelist= np.array(imagelist)
        label_Onehot = np_utils.to_categorical(label)
        np.save('image_input.npy',imagelist)
        np.save('label_finetune_restnet.npy',label_Onehot)
    seed=np.random.randint(0,100)
    np.random.seed(seed)
    np.random.shuffle(imagelist)
    np.random.seed(seed)
    np.random.shuffle(label_Onehot)
    train_data=imagelist[:9000]
    train_label=label_Onehot[:9000]
    test_data=imagelist[9000:]
    test_label=label_Onehot[9000:]
    #imagenet
    rest50 = ResNet50(weights='imagenet',include_top=True,input_shape=(224,224,3))
    x=rest50.layers[-1].input
    outputLayer = Dense(units=3,activation='softmax')(x)
    model=Model(inputs=rest50.input,outputs=outputLayer)
    #print(len(model.layers))
    mid = int(len(model.layers)*80/100)
    for i in range(mid):
        model.layers[i].trainable=False
    sgd = optimizers.gradient_descent_v2.SGD(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x=train_data,y=train_label,validation_split=0.5,batch_size=120,epochs=50,verbose=1)
    scores=model.evaluate(x=test_data,y=test_label)
    print(scores[1])
    model.save_weights('RestNet50.h5')
    return model

def VGG19FineTune(image_path,label):
    imagelist=[]
    label_Onehot = []
    if os.path.exists('image_input.npy') and os.path.exists('label_finetune.npy'):
        imagelist= np.load('image_input.npy')
        label_Onehot = np.load('label_finetune.npy')
    else:
        np.random.seed(999)
        np.random.shuffle(image_path)
        np.random.seed(999)
        np.random.shuffle(label)
        for path in image_path:
            img = Image.open(path)
            img = img.resize((224, 224))
            x = np.array(img)
            imagelist = imagelist+[preprocess_input(x)]
        imagelist= np.array(imagelist)
        label_Onehot = np_utils.to_categorical(label)
        np.save('image_input.npy',imagelist)
        np.save('label_finetune.npy',label_Onehot)
    train_data=imagelist[:8000]
    train_label=label_Onehot[:8000]
    test_data=imagelist[8000:]
    test_label=label_Onehot[8000:]

    vgg = VGG19(weights='imagenet',include_top=True,input_shape=(224,224,3))
    vggoutput = vgg.layers[-1].input
    outputLayer = Dense(units=3,activation='softmax')(vggoutput)
    sgd = optimizers.gradient_descent_v2.SGD(learning_rate=0.005)
    model=Model(inputs=vgg.input,outputs=outputLayer)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x=train_data,y=train_label,validation_data=[test_data,test_label],batch_size=32,epochs=50,verbose=1)
    model.save_weights('Vgg19.h5')
    return model

def extractFromBiLSTM(BiLSTM,train_data):
    model = Model(inputs=BiLSTM.input, outputs=BiLSTM.get_layer('multi_attention').output)
    feature=model.predict(train_data)
    feature=np.array(feature)
    np.save('text_data_extract_BiLSTM.npy',feature)
    return feature

def PreprocessImage(model,pathList):
    picFeatureList=[]
    for Item in pathList:
        feature=None
        img = Image.open(Item)
        img = img.resize((224, 224))
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        feature = np.resize(feature, (50, 41))  # 原本尺寸为2048=2048,改变形状为50X41=2050
        picFeatureList=picFeatureList+[feature]
    return np.array(picFeatureList)


def ProjectRun():
    text_data, image_path, label = readfile("MVSA")
    image_pretrained = None
    if os.path.exists('image_pretrained.npy') == False:
        model  = None
        if os.path.exists('RestNet50.h5') == False:
            model = RestNetFineTune(image_path.copy(),label.copy())
        else:
            rest = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
            x=rest.layers[-1].input
            outputLayer = Dense(units=3,activation='softmax')(x)
            model=Model(inputs=rest.input,outputs=outputLayer)
            model.load_weights('RestNet50.h5')
        image_pretrained=PreprocessImage(model,image_path)
        np.save('image_pretrained.npy',image_pretrained)
    else:
        image_pretrained = np.load('image_pretrained.npy')
    

    text_data_pretrained=None
    if os.path.exists('text_data_pretrained.npy')==False:
        text_data_pretrained=PreProcessWithBert(text_data)
    else:
        text_data_pretrained=np.load('text_data_pretrained.npy')
    print(type(text_data_pretrained))
    BiLSTM=None
    if os.path.exists('BiLSTM.h5') == False:
        model = CreateBiLSTMModel(text_data_pretrained)
        tp=text_data_pretrained.copy()
        lb=label.copy()
        np.random.seed(999)
        np.random.shuffle(tp)
        np.random.seed(999)
        np.random.shuffle(lb)
        label_Onehot = np_utils.to_categorical(lb)
        train_data = tp[:8000]
        test_data = tp[8000:]
        train_label = label_Onehot[:8000]
        test_label = label_Onehot[8000:]
        BiLSTM=TrainBiLSTMModel(model,train_data,train_label,test_data,test_label)
    else:
        BiLSTM =CreateBiLSTMModel(text_data_pretrained)
        BiLSTM.load_weights('BiLSTM.h5')
    
    text_data_extract_BiLSTM=None
    if os.path.exists('text_data_extract_BiLSTM.npy') ==False:
        text_data_extract_BiLSTM=extractFromBiLSTM(BiLSTM,text_data_pretrained)
    else:
        text_data_extract_BiLSTM=np.load('text_data_extract_BiLSTM.npy')

    concated_data=np.concatenate(( text_data_extract_BiLSTM ,image_pretrained),axis=2)
    np.random.seed(999)
    np.random.shuffle(concated_data)
    np.random.seed(999)
    np.random.shuffle(label)
    label_Onehot = np_utils.to_categorical(label)
    train_data=concated_data[:8000]
    test_data=concated_data[8000:]
    train_label =label_Onehot[:8000]
    test_label = label_Onehot[8000:]
    model=createModel(train_data)
    TrainModel(model,train_data,train_label,test_data,test_label)
    
#print(tf.test.is_gpu_available())
ProjectRun()
'''
rest = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
x=rest.layers[-1].input
outputLayer = Dense(units=3,activation='softmax')(x)
model=Model(inputs=rest.input,outputs=outputLayer)
sgd = optimizers.gradient_descent_v2.SGD(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.load_weights('RestNet50.h5')
test_data=np.load('image_input.npy')
test_label = np.load('label_finetune.npy')
score=model.evaluate(x=test_data[3000:6000],y=test_label[3000:6000])
print(score[1])
'''
#print(tf.test.is_gpu_available())
#model accuracy:69.68%
