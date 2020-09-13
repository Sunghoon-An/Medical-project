import sys
sys.path = ["/home/gruads/anaconda3/envs/clone/lib/python3.7/site-packages"]+sys.path
import numpy as np

import tensorflow as tf
import keras 
from keras import initializers
from keras.layers import Dense, Input, Activation, BatchNormalization, Dropout
from keras.layers import Lambda, concatenate, multiply
from keras.models import Model, Sequential, model_from_json

from keras.layers import Dropout
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_radam

from config import *


def self_attention(x):
    n = int(x.get_shape()[1])
    a = Dense(n, activation='sigmoid' #, name = f'attention_{i}'
              ,kernel_initializer='he_uniform')(x)
    return multiply([x, a])


def fc(x, n, batch = False, attention = False, acti = 'relu'):
    x = Dense(n, activation = acti, kernel_initializer= 'he_uniform')(x)
    x = BatchNormalization()(x) if batch else x
    x = self_attention(x) if attention else x
    return x

class AdversarialAutoEncoder():
    def __init__(self, input_dim):
        self.orignal_dim = input_dim
        self.input_shape = (self.orignal_dim,)
        self.intermediate_dim = 50
        self.latent_dim = 8
        self.optim = keras.optimizers.adam(1e-4)
        
    def build_encoder(self):
        input_layer = Input(shape=self.input_shape, name = 'input_layer')
        x = fc(input_layer, self.intermediate_dim, batch = False, attention = False
               , acti = 'relu')
        x = fc(x, self.intermediate_dim, batch = True, attention = True, acti = 'relu')
        x = fc(x, self.intermediate_dim, batch = True, attention = True, acti = 'relu')
        x = fc(x, self.latent_dim, batch = False, attention = False, acti = None)

        return Model(input_layer, x, name ='encoder')

    def build_decoder(self):
        latent_input = Input(shape=(self.latent_dim,), name = 'encode')
        x = fc(latent_input, self.intermediate_dim, batch = True, attention = True
               , acti = 'relu')
        x = fc(x, self.intermediate_dim, batch = True, attention = True, acti = 'relu')
        x = fc(x, self.intermediate_dim, batch = True, attention = True, acti = 'relu')
        output = fc(x, self.orignal_dim, batch = False, attention = False, acti = None)

        return Model(latent_input, output, name ='decoder')

    def build_discriminator(self):
        latent_input = Input(shape=(self.latent_dim,))

        x = fc(latent_input, self.intermediate_dim, batch = True, attention = True
               , acti = 'relu')
        x = fc(x, self.intermediate_dim, batch = True, attention = True, acti = 'relu')
        x = fc(x, self.intermediate_dim, batch = True, attention = True, acti = 'relu')
        output = fc(x, 1, batch = False, attention = False, acti = 'sigmoid')
        
        return Model(latent_input, output, name='discriminator')

    def train(self, x_train, x_test, batch_size, epoch, save_dir):
        batch_size = 64
        step = np.round(x_train.shape[0]/batch_size)
        val_step = np.round(x_test.shape[0]/batch_size)

        ae_input = Input(shape=self.input_shape, name = 'autoencoder_input')
        # generator_input = Input(shape=self.input_shape, name = 'generator_input')

        encoder = self.build_encoder()
        decoder = self.build_decoder()
        recon_out = decoder(encoder(ae_input))
        adversarial_ae = Model(ae_input, recon_out, name ='Adversarial_AutoEncoder')
        adversarial_ae.compile(optimizer = self.optim , loss = 'mean_squared_error')
        
        with open(os.path.join(save_dir, 'encoder_model.json'),'w') as f:
            f.write(encoder.to_json())
        
        discriminator = self.build_discriminator()
        discriminator.compile(optimizer = self.optim , loss = 'binary_crossentropy')
        
        with open(os.path.join(save_dir, 'discriminator_model.json'),'w') as f:
            f.write(discriminator.to_json())

        for e in range(epoch):
            recon_loss = []
            recon_val_loss = []
            disc_loss = []
            disc_val_loss = []
            for i in range(int(step-1)):
                batch = x_train[i*batch_size:(i+1)*batch_size]
                
                recon_loss.append(adversarial_ae.train_on_batch(batch, batch))

                fake_sample = encoder.predict(batch)
                b, f= fake_sample.shape

                discriminator_input = np.concatenate([fake_sample, np.random.randn(b, f)*2])
                discriminator_label = np.concatenate([np.zeros([b, 1]), np.ones([b, 1])])
                disc_loss.append(discriminator.train_on_batch(discriminator_input, discriminator_label))
                sys.stdout.write(f"\r{i}/{step-1} Done / epoch : {e+1} ")
                
            for j in range(int(val_step-1)):
                batch = x_test[j*batch_size:(j+1)*batch_size]
                recon_val_loss.append(adversarial_ae.test_on_batch(batch, batch))
                
                fake_sample = encoder.predict(batch)
                b, f= fake_sample.shape
                discriminator_input = np.concatenate([fake_sample, np.random.randn(b, f)])
                discriminator_label = np.concatenate([np.zeros([b, 1]), np.ones([b, 1])])
                disc_val_loss.append(discriminator.test_on_batch(discriminator_input, discriminator_label))
                sys.stdout.write(f"\r{i:05d}/{step-1} Done / epoch : {e+1:03d} ")
            
            recon_loss = np.mean(recon_loss)
            disc_loss = np.mean(disc_loss)
            recon_val_loss = np.mean(recon_val_loss)
            disc_val_loss = np.mean(disc_val_loss)
            
            
            if e == 0 :
                current_best = disc_val_loss
                encoder.save_weights(os.path.join(save_dir, 'weight', f'encoder_weight_{e+1:03d}_{recon_val_loss:2.4f}.h5') )
                discriminator.save_weights(os.path.join(save_dir, 'weight', f'discriminator_weight_{e+1:03d}_{disc_val_loss:2.4f}.h5') )
            else :    
                if current_best > disc_val_loss:
                    current_best = disc_val_loss
                    encoder.save_weights(os.path.join(save_dir, 'weight', f'encoder_weight_{e+1:03d}_{recon_val_loss:2.4f}.h5') )
                    discriminator.save_weights(os.path.join(save_dir, 'weight', f'discriminator_weight_{e+1:03d}_{disc_val_loss:2.4f}.h5') )

                elif current_best <= disc_val_loss:
                    pass
                else :
                    print( f"current_best : {current_best}, disc_val_loss : {disc_val_loss}")
            
            print("")
            print(f"recon_loss : {recon_loss:2.4f} disc_loss : {disc_loss:2.4f} recon_val_loss : {recon_val_loss:2.4f} disc_val_loss : {disc_val_loss:2.4f}")


def attention_autoencoder(input_dim, fs):
    # 25
    inputs = Input(shape=(input_dim,))
    
#     x1 = Lambda(lambda x : x[:,:(input_dim - fs)])(inputs)
#     x2 = Lambda(lambda x : x[:,-fs:])(inputs)
#     x2 = fc(x2, int(FEATURE_SIZE/2), batch = False, attention = False, acti = 'relu')    
#     x = concatenate([x1,x2], axis = -1)
    
    x = fc(inputs, input_dim, batch = True, attention = True, acti = 'relu')
    
    for i in range(3):
        x = fc(x, 32, batch = True, attention = True, acti = 'relu')
    
    x = fc(x, 32, batch = True, attention = True, acti = 'relu')
    x = fc(x, 8, batch = False, attention = False, acti = 'tanh')
    x = fc(x, 32, batch = True, attention = True, acti = 'relu')
    for i in range(3):
        x = fc(x, 32, batch = True, attention = True, acti = 'relu')
    
    out1 = fc(x, input_dim, batch = False, attention = False, acti = None)
    
    return Model(inputs, out1, name='autoencoder')


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]

    epsilon = K.random_normal(shape =(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def variational_autoencoder(orignal_dim, batch_size, fs):
    input_shape = (orignal_dim,)
    intermediate_dim = 32 
    latent_dim = 8

    inputs = Input(shape=input_shape, name = 'encoder_input')
    x1 = Lambda(lambda x : x[:,:(orignal_dim - fs)])(inputs)
    x2 = Lambda(lambda x : x[:,-fs:])(inputs)
    x2 = fc(x2, int(FEATURE_SIZE/2), batch = False, attention = False, acti = 'relu')
    x = concatenate([x1,x2], axis = -1)
    
    x = fc(x, intermediate_dim, batch = True, attention = True, acti = 'relu')
    for i in range(2):
        x = fc(x, intermediate_dim, batch = True, attention = True, acti = 'relu')
    
    x = fc(x, int(intermediate_dim/2), batch = True, attention = True, acti = 'relu')
    
    z_mean = Dense(latent_dim, activation = None, name='z_mean')(x)
    z_log_var = Dense(latent_dim, activation = None, name='z_log_var'
                      ,kernel_initializer = initializers.glorot_normal())(x)

    z = Lambda(sampling, output_shape = (latent_dim,), name ='z')([z_mean, z_log_var])

    #encoder = Model(inputs, [z_mean, z_log_var, z], name ='encoder')
    #latent_input = Input(shape=(latent_dim,), name = 'z_sampling')
    
    x = fc(z, int(intermediate_dim/2), batch = True, attention = True, acti = 'relu') 
    for i in range(2):
        x = fc(x, intermediate_dim, batch = True, attention = True, acti = 'relu')
    x = fc(x, intermediate_dim, batch = True, attention = True, acti = 'relu')
    output = fc(x, orignal_dim, batch = False, attention = False, acti = None)
    
    # output = Dense(orignal_dim, activation = None)(x)
    # decoder = Model(latent_input, output, name ='decoder')
    # output = decoder(encoder(inputs)[2])

    return Model(inputs, output, name='vae')
    
    
if __name__ == "__main__":
    FEATURE_SIZE = 10
    # model = creat_autoencoder(1200)
    model = variational_autoencoder(25,32)
    model.summary()
    