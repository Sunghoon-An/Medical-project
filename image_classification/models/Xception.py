# -*- coding: utf-8 -*-

from models.layers import *
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np

def reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    '''
    Description :
        주어진 Kernel size보다 작은 input을 가지고 있을 경우, 해당 input의 크기로 Kernel size를 재조정
        
    Arguments :
        - input_tensor : 입력 텐서
        - kernel_size : 사용할 kernel size
        
    Output :
        재조정된 Kernel size
    '''
    shape = input_tensor.get_shape().as_list()
    
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])]
        
    return kernel_size_out
    
class create():
    '''
    Description :
        Xception Model을 생성하기 위한 모델 정의 
        Xception: Deep Learning with Depthwise Separable Convolutions 
        [Link] https://arxiv.org/pdf/1610.02357.pdf
    
    Attributes :  ## V : 변수 , F : 함수 ##
        - [V] graph : Xception Model graph 정의
        - [V] num_output : Classification 할 class 갯수
        - [V] reduction_ratio : Squeeze Exciation Layer 사용시 Squeeze할 비율
        - [V] enable_SE : Squeeze Exciation Layer 사용 여부
        - [V] data_format : 입력 데이터 shape 중에서 어느 차원이 channel을 의미하는지 명시
        - [V] sess : 모델의 graph와 gpu 사용 옵션 설정이 들어간 tensorflow session
        - [V] x : 입력 이미지 변수 (placeholder)
        - [V] y : 정답 변수 (placeholder)
        - [V] is_train : 현재 학습 여부 변수 (placeholder)
        - [V] lr : 학습률 (placeholder)
        - [V] loss : 손실함수 - sparse_softmax_cross_entropy (placeholder)
        - [V] train_op : 학습용 Optimizer
        - [V] logits : network의 각각의 Class에 대한 logistic regression
        - [V] prob : 각 클래스에 대한 score
    
        - [F]create_model : 실제 xception 모델을 생성
        - [F]set_op : Optimizer 설정 함수
    '''
    def __init__(self, data_shape, num_output, reduction_ratio=4, data_format='channels_last', batch_size=None, 
                 enable_SE = False, gpu_memory_fraction=None, optimizer_type='adam', phase='train'):
        '''
        Description :
            Xception Model을 생성 및 초기화를 진행
            
        Arguments : 
            - data_shape : 입력할 데이터의 shape
            - num_output : Classification 할 class 갯수
            - reduction_ratio (기본값 : 4) : Squeeze Exciation Layer 사용시 Squeeze할 비율
            - data_format (기본값 : channels_last) : 입력 데이터 shape 중에서 어느 차원이 channel을 의미하는지 명시
            - batch_size (기본값 : None) : batch_size
            - enable_SE (기본값 : False) : Squeeze Exciation Layer 사용 여부
            - gpu_memory_fraction (기본값 : None) : GPU 사용량 제한
            - optimizer_type (기본값 : adam) : 학습에 사용할 Optimizer 정의
            - phase (기본값 : train) : 현재 모델이 train을 위해 사용하는지 inference만을 위해 사용하는지 명시
            
        '''
        
        if phase not in ['train', 'inference'] : 
            raise  ValueError("phase must be 'train' or 'inference'.")
            
        # 기본 정의
        self.graph = tf.get_default_graph()
        self.num_output = num_output
        self.reduction_ratio = reduction_ratio
        self.enable_SE = enable_SE
        self.data_format = data_format
        
        # GPU 사용량 제한 및 설정
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        
        # GPU사용량 설정과 사용할 Graph를 Session에 정의
        self.sess = tf.Session(config=config, graph=self.graph)
        
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, (batch_size,) + data_shape, name='input_images')
            self.y = tf.placeholder(tf.int32, [batch_size,])
            self.is_train = tf.placeholder(tf.bool)
            self.__create_model() 
            self.loss = tf.losses.sparse_softmax_cross_entropy(self.y, self.logits)
            
            # 만약 학습일 경우, 학습을 위한 learning rate와 Optimizer 정의 후 전체 모델의 변수들을 초기화
            if phase=='train':
                self.lr = tf.placeholder(tf.float32, name="lr")

                # 학습에 사용할 Optimizer 정의 및 변수 초기화
                self.sess.run(tf.global_variables_initializer())
                self.train_op = self.__set_op(self.loss, self.lr, optimizer_type)
                uninit_vars = [v for v in tf.global_variables()
                              if not tf.is_variable_initialized(v).eval(session=self.sess)]
                self.sess.run(tf.variables_initializer(uninit_vars))

    def __create_model(self):
        '''
        Description :
            Xception Model을 실제로 생성. 
        '''
        # [ Entry Flow ] - Stage1
        data = batch_norm(self.x, self.is_train)
        
        data = conv(data, 32, 3, 2, padding="VALID", conv_name="block1-conv", bn_name="block1-bn",
                    use_bias=False, bn=True, act=True, is_train=self.is_train)
        data = conv(data, 64, 3, 1, padding="VALID", conv_name="block2-conv", bn_name="block2-bn",
                    use_bias=False, bn=True, act=True, is_train=self.is_train)
        residual = conv(data, 128, 1, 2, padding="VALID", conv_name="block3-res_conv", bn_name="block3-res_bn",
                    use_bias=False, bn=True, act=False, is_train=self.is_train)
        
        # [ Entry Flow ] - Stage2
        data = separable_conv(data, 128, self.is_train, act=False, name='block4-dws_conv')
        data = separable_conv(data, 128, self.is_train, name='block5-dws_conv')
        data = max_pooling(data)
        data = tf.add(data, residual)
        if self.enable_SE : data = squeeze_excitation_layer(data, self.reduction_ratio)
        residual = conv(data, 256, 1, 2, padding="VALID", conv_name="block6-res_conv", bn_name="block6_res_bn",
                    use_bias=False, bn=True, act=False, is_train=self.is_train)
        
        # [ Entry Flow ] - Stage3
        data = separable_conv(data, 256, self.is_train, name='block7-dws_conv')
        data = separable_conv(data, 256, self.is_train, name='block8-dws_conv')
        data = max_pooling(data)
        data = tf.add(data, residual)
        if self.enable_SE : data = squeeze_excitation_layer(data, self.reduction_ratio)
        residual = conv(data, 728, 1, 2, padding="VALID", conv_name="block9-res_conv", bn_name="block9_res_bn",
                    use_bias=False, bn=True, act=False, is_train=self.is_train)
        
        # [ Entry Flow ] - Stage 4
        data = separable_conv(data, 728, self.is_train, name='block10-dws_conv')
        data = separable_conv(data, 728, self.is_train, name='block11-dws_conv')
        data = max_pooling(data)
        data = tf.add(data, residual)
        if self.enable_SE : data = squeeze_excitation_layer(data, self.reduction_ratio)
        
        # [ Middle Flow ] - Stage 5
        block_num = 12
        SE_num = 4
        for _ in range(8):
            residual = data
            
            for _ in range(3):
                data = separable_conv(data, 728, self.is_train, name='block{}-dws_conv'.format(block_num))
                block_num+=1
            data = tf.add(data, residual)
            if self.enable_SE : 
                data = squeeze_excitation_layer(data, self.reduction_ratio)
                SE_num+=1
            
        # [ Exit Flow ] - Stage 6
        residual = conv(data, 1024, 1, 2, conv_name="block37-res_conv", bn_name="block37_res_bn",
                    use_bias=False, bn=True, act=False, is_train=self.is_train, padding="SAME")
        data = separable_conv(data, 728, self.is_train, name='block38-dws_conv')
        data = separable_conv(data, 1024, self.is_train, name='block39-dws_conv')
        data = max_pooling(data)
        data = tf.add(data, residual)
        if self.enable_SE : data = squeeze_excitation_layer(data, self.reduction_ratio)
        
        # [ Exit Flow ] - Stage 7
        data = separable_conv(data, 1536, self.is_train, act=False, name='block40-dws_conv')
        data = separable_conv(data, 2048, self.is_train, name='block41-dws_conv')
        data = tf.nn.relu(data)
        
        if self.enable_SE : data = squeeze_excitation_layer(data, self.reduction_ratio)
        
        if self.data_format == 'channels_first':
            channels_last_inputs = tf.transpose(data, [0, 2, 3, 1])
        else:
            channels_last_inputs = data
        
        # 만약 Kernel size가 10,10보다 작은 input을 가지고 있을 경우, 해당 input의 크기로 Kernel size를 재조정
        reduced = reduced_kernel_size_for_small_input(channels_last_inputs, [10, 10])
        data = tf.layers.average_pooling2d(data, pool_size = reduced, strides = 1, 
                                           padding='valid', data_format=self.data_format, name='avg_pool')

        data = global_avg_pooling(data)
        
        # Logistic Regression for Classification
        self.logits = fc(data, self.num_output, name='FC', relu=False, bn=False, is_train=self.is_train)
        # Classification Score
        self.prob = tf.nn.sigmoid(self.logits)
        
    def __set_op(self, loss_op, learning_rate, optimizer_type="adam"):
        '''
        Description :
            학습에 사용할 Optimizer 생성
        
        Arguments :
            - loss_op : 최적화할 손실 함수
            - learning_rate : 사용할 학습률
            - optimizer_type (기본값 : adam) : 사용할 Optimizer 종류 {'adam', 'adagrad', 'sgd', 'momentum', 'adadelta'}
        
        Output :
            주어진 Loss 함수에 대해 최적화를 진행할 Optimizer
        '''
        with self.graph.as_default():
            if optimizer_type=="adam":
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif optimizer_type == "adagrad":
                optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.0001)
            elif optimizer_type == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            elif optimizer_type == "momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
            elif optimizer_type == "adadelta":
                optimizer = tf.train.AdadeltaOptimizer(learning_rate,rho=0.95,epsilon=1e-09)
            else : raise ValueError("{} optimizer doesn't exist.".format(optimizer_type))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss_op)

        return train_op
