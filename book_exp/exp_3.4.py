from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


def seq_encode(data):
    """
    将评论反解码成对应的英文单词
    :param data:
    :return:void
    """
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ''.join(reverse_word_index.get(i-3, '?') for i in data[0])
    print(decoded_review)


def vectorize_sequences(sequences, dimension=10000):
    """
    :param sequences:
    :param dimension:
    将整数序列用one-hot方法转换成二进制矩阵
    :return:
    """
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1
    return results


# 数据加载
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 数据以及标签格式化处理
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 预留验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 模型定义
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 编译模型:选定优化器与损失函数
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 从history提取信息
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']
epochs = range(1, len(loss_values) + 1)

# 绘制图像
# plt.clf() # 清空图像
plt.plot(epochs, loss_values, 'bo', label='Training_loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation_loss')
plt.plot(epochs, acc, 'ro', label='Training_acc')
plt.plot(epochs, val_acc, 'r', label='Validation_acc')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

# 测试集结果
results = model.evaluate(x_test, y_test)
print("resuls of testing data:", results)

# 用于预测
predict = model.predict(x_test)
print("prediction of testing data:", predict)
