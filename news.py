from keras.datasets import  reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import  models
from keras import  layers
import copy
(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)
def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results
x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)

def to_one_hot(labels,dimension=46):
    results=np.zeros(((len(labels),dimension)))
    for i,label in enumerate(labels):
        results[i,label]=1
    return results
one_hot_train_labels=to_one_hot(train_labels)
one_hot_test_labels=to_one_hot(test_labels)

one_hot_train_labels=to_categorical(train_labels)
one_hot_test_labels=to_categorical(test_labels)

model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

# 编译模型
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# 验证集
x_val=x_train[:1000]
particial_x_train=x_train[1000:]

y_val=one_hot_train_labels[:1000]
particial_y_train=one_hot_train_labels[1000:]

history=model.fit(particial_x_train,particial_y_train,epochs=9,batch_size=512,validation_data=(x_val,y_val))
results=model.evaluate(x_test,one_hot_test_labels)

test_labels_copy=copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array=np.array(test_labels)==np.array(test_labels_copy)
print(float(np.sum(hits_array))/len(test_labels))





