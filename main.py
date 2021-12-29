# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

from keras import  models
from keras import  layers
from keras.datasets import mnist
from keras.utils import  np_utils

# 加载数据
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
# 网络架构
network=models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
# 编译
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# 准备图像数据
train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255

# 准备标签
train_labels=np_utils.to_categorical(train_labels)
test_labels=np_utils.to_categorical(test_labels)

# 拟合模型
network.fit(train_images,train_labels,epochs=5,batch_size=128)

#测试数据
test_loss,test_acc=network.evaluate(test_images,test_labels)
print('test_acc:',test_acc)









# 按间距中的绿色按钮以运行脚本。


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
