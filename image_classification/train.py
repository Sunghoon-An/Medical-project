# -*- coding: utf-8 -*-
``` 모든 데이터와 Root는 보완상 $로 표시```

from scripts import dataset
from scripts.tools import *
from models import Xception
from sklearn import metrics
import tensorflow as tf
import warnings, gc, os ; warnings.filterwarnings('ignore')
    
# 사용할 GPU 설정
gpu_Num = '0'
set_GPU(gpu_Num)

# 호출할 전처리 데이터 및 학습 결과 저장할 위치
dataset_path = '/$$$'
save_path = '/$$$/$$$'
if not os.path.exists(save_path) :
    os.makedirs(save_path)

# Xception 모델 
tf.reset_default_graph()
net = Xception.create(data_shape=(256,512,1), num_output=2, reduction_ratio=4, optimizer_type='adadelta', enable_SE=True)

# 전처리 데이터 호출
cls_data = dataset.classification(dataset_root=dataset_path, restore=True)

# Training과 Validation을 평가하기위해 Batch로 쪼개 두었던 Label 통합
train_labels = np.concatenate(cls_data.batch_set['train']['label'], axis=0)
valid_labels = np.concatenate(cls_data.batch_set['valid']['label'], axis=0)

# Epoch 설정
num_epochs = 100

# 최대치, 최소치 저장
lowest_loss=None
highest_auc=None
highest_acc=None

# 학습 Epoch 시작
with tf.device('/gpu:0'):
    # 모델 파라미터 초기화
    net.sess.run(tf.global_variables_initializer())
    
    # 학습 모델 저장 설정
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(save_path, net.sess.graph)

    # Epoch 
    for epoch in range(num_epochs):

        train_pred = []
        train_prob = []
        train_loss = 0.
        train_count = 0

        val_pred = []
        val_prob = []
        val_loss = 0.
        val_count = 0

        # Iteration
        for i in range(len(cls_data.batch_set['train']['image'])):
            prob, cost, _ = net.sess.run([net.prob, net.loss, net.train_op], 
                                         feed_dict={net.x: cls_data.batch_set['train']['image'][i][:,:,:,np.newaxis],
                                                    net.y: cls_data.batch_set['train']['label'][i], 
                                                    net.lr : 1.,
                                                    net.is_train: True})

            train_pred+=list(prob.argmax(-1))
            train_prob+=list(prob[:,1])
            train_loss += cost
            train_count += 1

        for i in range(len(cls_data.batch_set['valid']['image'])):
            prob, cost = net.sess.run([net.prob, net.loss], 
                                      feed_dict={net.x: cls_data.batch_set['valid']['image'][i][:,:,:,np.newaxis],
                                                 net.y: cls_data.batch_set['valid']['label'][i],
                                                 net.is_train: False})

            val_pred+=list(prob.argmax(-1))
            val_prob+=list(prob[:,1])
            val_loss += cost
            val_count += 1

        # Train - AUROC, Sensitivity, Specificity, Accuracy
        tn, fp, fn, tp = metrics.confusion_matrix(train_labels, train_pred).ravel()

        train_auc = metrics.roc_auc_score(train_labels, train_prob) # AUROC
        train_acc = metrics.accuracy_score(train_labels, train_pred) # Accuracy
        train_sensitivity = tp / (tp+fn) # Sensitivity
        train_specificity = tn / (tn+fp) # Specificity

        # Valid - AUC, Sensitivity, Specificity, Accuracy
        tn, fp, fn, tp = metrics.confusion_matrix(valid_labels, val_pred).ravel()

        valid_auc = metrics.roc_auc_score(valid_labels, val_prob) # AUROC
        valid_acc = metrics.accuracy_score(valid_labels,val_pred) # Accuracy
        valid_sensitivity = tp / (tp+fn) # Sensitivity
        valid_specitivity = tn / (tn+fp) # Specificity

        # 결과 출력
        Result = "[Epochs : "+str(epoch+1)+" ]"+ \
                 " Train - AUC : "+str(round(train_auc,5))+ \
                 " Train - Accuracy : "+str(round(train_acc,5)) + \
                 " Train - Sensitivity : "+str(round(train_sensitivity,5))+ \
                 " Train - Specitivity : "+str(round(train_specitivity,5))+ \
                 " Train - Loss : "+str(round(train_loss/train_count,5) if train_count !=0 else 0)+ \
                 " Val - AUC : "+str(round(valid_auc,5))+ \
                 " Val - Accuracy : "+str(round(valid_acc,5)) + \
                 " Val - Sensitivity : "+str(round(valid_sensitivity,5))+ \
                 " Val - Specitivity : "+str(round(valid_specitivity,5))+ \
                 " Val - Loss : "+str(round(val_loss/val_count,5) if val_count !=0 else 0)
        print(Result)
        
        # Tensorboard에 Measurement 기록
        summ = tf.Summary()
        summ.value.add(tag='Validation_loss', simple_value=val_loss/val_count if val_count !=0 else 0)
        summ.value.add(tag='Validation_AUC', simple_value=valid_auc)
        summ.value.add(tag='Validation_Accuracy', simple_value=valid_acc)
        summ.value.add(tag='Validation_Sensitivity', simple_value=valid_sensitivity)
        summ.value.add(tag='Validation_Specitivity', simple_value=valid_specitivity)

        summ.value.add(tag='Train_loss', simple_value=train_loss/train_count if train_count !=0 else 0)
        summ.value.add(tag='Train_AUC', simple_value=train_auc)
        summ.value.add(tag='Train_Accuracy', simple_value=train_acc)
        summ.value.add(tag='Train_Sensitivity', simple_value=train_sensitivity)
        summ.value.add(tag='Train_Specitivity', simple_value=train_specitivity)
        summary_writer.add_summary(summ,epoch)

        # Epoch 2회부터 Validation 성능에 따라 학습 결과 모델 저장 
        if epoch > 0:
            # 가장 Loss가 낮을 때 저장
            if lowest_loss == None or lowest_loss > val_loss/val_count :
                lowest_loss = val_loss/val_count
                saver.save(net.sess, save_path+"Xception_lowest_loss.ckpt")
            # 가장 AUROC가 높을 때 저장
            if highest_auc == None or highest_auc < valid_auc :
                highest_auc = valid_auc
                saver.save(net.sess, save_path+"Xception_highest_auc.ckpt")
            # 가장 Accuracy가 높을 때 저장
            if highest_acc == None or highest_acc < valid_acc :
                highest_acc = valid_acc
                saver.save(net.sess, save_path+"Xception_highest_acc.ckpt")
