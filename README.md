## CNN Classficition Tensorflow

>ʹ��Tensorflow����һ�����������ķ�������

ʶ��ͼƬ�е����Ʊ�ʾ�����֣�0-5

### ���ݼ�
- ѵ��������Ϊ $1080$��
- ���Լ�����Ϊ $120$ ��
- ÿ��ͼƬ�ĳߴ�Ϊ $64\times 64\times 3$


### ʵ�ֹ���



- ����㣨Conv2D��: stride 1, padding is 'SAME'
- �����Լ������Relu��
- �ػ��㣨Max Pool��: 8 by 8 filter, 8 by 8 stride, padding is 'SAME'
<br />
- ����㣨Conv2D��: stride 1, padding is 'SAME'
- �����Լ������Relu��
- �ػ��㣨Max Pool��: 4 by 4 filter, 4 by 4 stride, padding is 'SAME'
<br />
- Flatten��������ȫ����ǰԤ����
- ȫ���Ӳ㣨FullyConnected layer��

ģ��ʹ�� $Mini-batch$ �ݶ��½��� $Adam$ �Ż�������ѧϰ����С����ʧ


### ����ṹ
```python
def creat_placeholders(n_H0, n_W0, n_C0, n_y):
    '''
    ����ռλ��������ͼƬ�ĳߴ� + ������������
    '''
    return X, Y
    
def initialize_parameters():
    '''
    ��ʼ��Ȩ�ز���   W1:[4,4,3,8]    W2:[2,2,8,16]
    '''
    return parameters

def forward_propagation(X, parameters):
    ''''''
	return Z3

def compute_cost(Z3, Y):
    '''
    Z3 - ǰ�д����������
    Y - ��ʵ��ǩ������ռλ������Z3��״��ͬ
    '''
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, 
			num_epochs=100, minibatch_size=64, print_cost=True):
	'''ģ�ͣ���������ģ��'''
	return train_accuracy, test_accuracy, parameters
```

### ���н��

- ���Լ���׼ȷ��ԼΪ��$94.5 \%$
- ���Լ���׼ȷ��ԼΪ��$81.7 \%$