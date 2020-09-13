# -*- coding: utf-8 -*-

import tensorflow as tf
from tflearn.layers.conv import global_avg_pool

def conv(data, depth, ksize, ssize, padding, use_bias, conv_name=None, bn_name=None, bn=False, act=True, is_train=None):
    """
    Description :
        Convolution Layer 구현
        
    Arguments :
        - data : Tensor Type의 입력 데이터 (B, W, H, C)
        - ksize : Kernel Size
        - depth : feature map 갯수
        - ssize : stride size
        - padding : 사용할 padding type {'SAME', 'VALID'}
        - use_bias : Bias 사용 여부
        - name (기본값 : False) : Layer에 사용할 이름
        - bn (기본값 : True) : batch normalization 사용 여부
        - act (기본값 : True) : 활성화함수 ReLU 사용여부
        - is_train (기본값 : None) : 현재 학습 상태 여부
        
    Output :
        Tensor Type output (batch size, width, height , depth)
    """
    
    output = tf.layers.conv2d(data, kernel_size=ksize, filters=depth,
                              strides=(ssize,ssize),
                              padding=padding.upper(),
                              name=conv_name,use_bias=use_bias,
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    # Batch normalization
    if bn : 
        output = batch_norm(output, is_train, bn_name)
    # ReLU
    if act : 
        output = tf.nn.relu(output)
    return output

def separable_conv(data, depth, ksize=3, ssize=1, padding, use_bias, data_format='channels_last',
                   name=None, bn=True, act=True, is_train=None):
    """
    Description :
        Separable Convolution Layer 구현
        
    Arguments :
        - data : Tensor Type의 입력 데이터 (B, W, H, C)
        - ksize : Kernel Size
        - depth : feature map 갯수
        - ssize : stride size
        - padding : 사용할 padding type {'SAME', 'VALID'}
        - use_bias : Bias 사용 여부
        - data_format (기본값 : 'channels_last') : 입력 데이터의 차원 순서 명시. {'channels_last' : (batch, height, width, channels)
                                                                          'channels_first' : (batch, channels, height, width)}
        - name (기본값 : False) : Layer에 사용할 이름
        - bn (기본값 : True) : batch normalization 사용 여부
        - act (기본값 : True) : 활성화함수 ReLU 사용여부
        - is_train (기본값 : None) : 현재 학습 상태 여부
        
    Output :
        Tensor Type output (B, W, H , depth)
    """
    # ReLU
    if act : 
        data = tf.nn.relu(data)
            
    # Separable Convolution
    data = tf.layers.separable_conv2d(data, depth, ksize,
                            strides=(ssize,ssize), padding='SAME',
                            data_format = data_format,
                            activation = None , use_bias = False,
                            depthwise_initializer = tf.contrib.layers.xavier_initializer(),
                            pointwise_initializer = tf.contrib.layers.xavier_initializer(),
                            bias_initializer = tf.zeros_initializer(),
                            name=name, reuse=None)
    # Batch normalization
    if bn :
        data = batch_norm(data, is_train)
    return data

def max_pooling(data, ksize=3, ssize=2, padding="SAME", name=None):
    """
    Description :
        Max Pooling 구현
        
    Arguments :
        - data : Tensor Type의 입력 데이터
        - ksize (기본값 : 3) : Kernel Size
        - ssize (기본값 : 2) : stride size
        - padding (기본값 : "SAME") : 사용할 padding type {'SAME', 'VALID'}
        - name (기본값 : None) : 해당 함수에 사용할 이름
        
    Output :
        Tensor Type output
    """
    return tf.nn.max_pool(data, ksize=[1,ksize,ksize,1], strides=[1,ssize,ssize,1], padding=padding, name=name)
    
def global_avg_pooling(data, name=None):
     """
    Description :
        Global Average Pooling (channel wise)구현
        
    Arguments :
        - data : Tensor Type의 입력 데이터
        - name (기본값 : None) : 해당 함수에 사용할 이름
        
    Output :
        Tensor Type output
    """
    return global_avg_pool(data)

def batch_norm(data, is_train, name=None, bn_axis=-1, USE_FUSED_BN = False, BN_EPSILON = 0.001, BN_MOMENTUM = 0.99):
    """
    Description :
        Batch Normalization 구현
        
    Arguments :
        - data : Tensor Type의 입력 데이터
        - is_train : 현재 학습 상태 여부
        - name (기본값 : 'Batch_Norm') : 해당 함수에 사용할 이름
        - data_format (기본값 : 'channels_last') : 입력 데이터의 차원 순서 명시. {'channels_last' : (batch, height, width, channels)
                                                                          'channels_first' : (batch, channels, height, width)}
        - USE_FUSED_BN (기본값 : True ) : {True, None} 일 경우, 더 빠른 알고리즘으로 사용, {False}시에는 시스템 권장 사항 사용
        - BN_EPSILON (기본값 : 0.001 ) : 0 으로 나누어지는 것을 방지하기위해 0 대신 사용할 숫자
        - BN_MOMENTUM (기본값 : 0.99 ) : moving average에 대한 모멘텀
        
    Output :
        Tensor Type output
    """
    # 현재 학습 상태를 입력하지 않으면 에러
    if is_train is None:
        raise ValueError('Phase should be declared "True" for train or "False" for test when you activate batch normalization.')
    # Channel 축 인덱스 정의
    bn_axis = -1 if data_format == 'channels_last' else 1
    
    # Batch Normalization
    return tf.layers.batch_normalization(data, training=is_train, name=name, axis=bn_axis,
                                         momentum=BN_MOMENTUM, epsilon=BN_EPSILON,fused=USE_FUSED_BN)

def fc(data, num_out, name=None, relu=True, bn=True, is_train=None):
    """
    Description :
        Fully Connected Layer 구현
        
    Arguments :
        - data : Tensor Type의 입력 데이터
        - num_out : 출력할 노드의 개수
        - name (기본값 : 'None') : 해당 함수에 사용할 이름
        - relu (기본값 : True) : 활성화함수 ReLU 사용여부
        - bn (기본값 : True) : batch normalization 사용 여부
        - is_train : 현재 학습 상태 여부
        
    Output :
        Tensor Type output
    """
    # Fully connected Layer
    output = tf.layers.dense(inputs=data, use_bias=True, units=num_out)
    
    # Batch normalization
    if bn : 
        output = batch_norm(output, is_train)
        
    # ReLU
    if relu : 
        output = tf.nn.relu(output)

    return output
        
def squeeze_excitation_layer(data, ratio):
    """
    Description :
        Squeeze Exciation (Channel Responce) 구현
        
    Arguments :
        - data : Tensor Type의 입력 데이터
        - ratio : Squeeze 할 때 압출을 할 비율
        
    Output :
        Tensor Type output (입력 데이터와 같은 shape)
    """
    c = data.get_shape().as_list()[-1]
    squeeze = global_avg_pooling(data)

    excitation = fc(squeeze, c/ratio, bn=False, relu=True)
    excitation = fc(excitation, c, bn=False, relu=False)
    excitation = tf.nn.sigmoid(excitation)

    excitation = tf.reshape(excitation, [-1,1,1,c])

    scale = data * excitation

    return scale