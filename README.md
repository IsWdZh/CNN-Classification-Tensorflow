# CNN Classficition Tensorflow

>使用Tensorflow构建一个卷积神经网络的分类问题

识别图片中的手势表示的数字：0-5

### 数据集
- 训练集数据为 1080个
- 测试集数据为 120 个
- 每张图片的尺寸为 64 × 64 × 3


### 实现过程

- 卷积层（Conv2D）: stride 1, padding is 'SAME'
- 非线性激活函数（Relu）
- 池化层（Max Pool）: 8 by 8 filter, 8 by 8 stride, padding is 'SAME'
<br />

- 卷积层（Conv2D）: stride 1, padding is 'SAME'
- 非线性激活函数（Relu）
- 池化层（Max Pool）: 4 by 4 filter, 4 by 4 stride, padding is 'SAME'
<br />

- Flatten向量化：全连接前预处理
- 全连接层（FullyConnected layer）






模型使用 Mini-batch 梯度下降和 Adam 优化，加速学习，最小化损失


### 程序结构
```python
def creat_placeholders(n_H0, n_W0, n_C0, n_y):
    '''
    创建占位符（输入图片的尺寸 + 分类器个数）
    '''
    return X, Y
    
def initialize_parameters():
    '''
    初始化权重参数   W1:[4,4,3,8]    W2:[2,2,8,16]
    '''
    return parameters

def forward_propagation(X, parameters):
    ''''''
	return Z3

def compute_cost(Z3, Y):
    '''
    Z3 - 前行传播最后的输出
    Y - 真实标签向量的占位符，和Z3形状相同
    '''
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, 
			num_epochs=100, minibatch_size=64, print_cost=True):
	'''模型，整个各个模块'''
	return train_accuracy, test_accuracy, parameters
```

### 运行结果

- 测试集的准确率约为：94.5%
- 测试集的准确率约为：81.7%
