import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255
X_test = X_test_orig/255
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

# print(X_train.shape)      # (1080, 64, 64, 3)
# print(X_test.shape)       # (120, 64, 64, 3)


def creat_placeholders(n_H0, n_W0, n_C0, n_y):
    '''
    创建占位符（输入图片的尺寸 + 分类器个数）
    '''
    X = tf.placeholder(shape=[None, n_H0, n_W0, n_C0], dtype=tf.float32)    # None：不限定训练实例数量
    Y = tf.placeholder(shape=[None, n_y], dtype=tf.float32)
    return X, Y

# X, Y = creat_placeholders(64,64,3,6)
# print(X,'\n',Y)

def initialize_parameters():
    '''
    初始化权重参数   W1:[4,4,3,8]    W2:[2,2,8,16]
    '''
    W1 = tf.get_variable('W1',[4,4,3,8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2',[2,2,8,16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"W1": W1,
                  "W2": W2}
    return parameters

# tf.reset_default_graph()
# with tf.Session() as sess_test:
#     parameters = initialize_parameters()
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#     print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
#     print("W2 = " + str(parameters["W2"].eval()[1,1,1]))

def forward_propagation(X, parameters):
    '''
    - Cnv2D: stride 1, padding is 'SAME'
    - Relu
    - Max Pool: 8 by 8 filter, 8 by 8 stride, padding is 'SAME'

    - Conv2D: stride 1, padding is 'SAME'
    - Relu
    - Max pool: 4 by 4 filter, 4 by 4 stride, padding is 'SAME'

    -Flatten
    -FullyConnected layer
    '''
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')     # strides  (m, n_H_prev, n_W_prev, n_C_prev)
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')   # ksize=[1,f,f,1]  stride=[1,s,s,1]

    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

    F = tf.contrib.layers.flatten(P2)      # 将每个例子展开为1维 [batch_size, k]
    Z3 = tf.contrib.layers.fully_connected(F, 6, activation_fn=None)    # 输出6个特征（神经元） 即下一层单元数  激活函数

    return Z3


def compute_cost(Z3, Y):
    '''
    Z3 - 前行传播最后的输出
    Y - 真实标签向量的占位符，和Z3形状相同
    '''
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, num_epochs=100, minibatch_size=64, print_cost=True):
    '''
    num_epochs: 代数，即第几次遍历数据集（学习率衰减）
    minibatch_size; minibatch梯度下降法
    '''
    ops.reset_default_graph()    # 不覆盖变量的情况下重新运行模型
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    X, Y = creat_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)

    # Adam优化算法（将Momentum和RMSprop结合起来）
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m/minibatch_size)   # minibatch 数量（train中)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                useless, temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches

            if print_cost == True and epoch % 5 ==0:
                print("第 %i 代后，损失为：%f"%(epoch,minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(np.squeeze(costs))
        plt.ylabel('损失')
        plt.xlabel('迭代次数（*10）')
        plt.title("学习率 = " + str(learning_rate))
        plt.show()

        predict_op = tf.argmax(Z3, 1)      # 返回最大数值的下标 array 行
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))     # 比较相同维度矩阵相同位置的元素
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print(train_accuracy)
        print(test_accuracy)

        return train_accuracy, test_accuracy, parameters

_, _, parameters = model(X_train, Y_train, X_test, Y_test)











