import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Dense, Input, Activation, BatchNormalization, Dropout, Reshape
from tensorflow.keras.layers import Lambda, concatenate, multiply, Conv2D, Flatten, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.layers import Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import mse, binary_crossentropy

print(f"tensorflow version {tf.__version__}")


class VariationalOversampler():
    
    def __init__(self, latent_dim = 5):
        self.latent_dim = latent_dim
    
    
    def _sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]

        epsilon = K.random_normal(shape =(batch, dim), mean=0.0, stddev=1.0)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    
    def _cnn_variational_autoencoder(self, orignal_dim):

        inputs = Input(shape= orignal_dim, name = 'encoder_input')
        x = Conv2D(64, 3, padding='same', activation='relu')(inputs)
        x = Conv2D(32, 3, padding='same', activation='relu')(x)
        x = Conv2D(3, 3, padding='same', activation='relu')(x)
        shape_bf_flatten = x.shape[1:]
        x = Flatten()(x)
        x = Dense(np.prod(shape_bf_flatten), activation = 'relu', kernel_initializer= 'he_uniform')(x)
        x = Dense(1000, activation = 'relu', kernel_initializer= 'he_uniform')(x)

        x = Dense(self.latent_dim, activation = 'relu', kernel_initializer= 'he_uniform')(x)
        z_mean = Dense(self.latent_dim, activation = None, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, activation = None, name='z_log_var'
                          ,kernel_initializer = initializers.glorot_normal())(x)
        z = Lambda(self._sampling, output_shape = (self.latent_dim,), name ='z')([z_mean, z_log_var])

        encoder = Model(inputs, [z_mean, z_log_var, z], name ='encoder')

        latent_input = Input(shape=(self.latent_dim,), name = 'z_sampling')
        x = Dense(1000, activation = 'relu', kernel_initializer= 'he_uniform')(latent_input)
        x = Dense(np.prod(shape_bf_flatten), activation = 'relu', kernel_initializer= 'he_uniform')(x)
        x = Reshape(shape_bf_flatten)(x)
        x = Conv2D(3, 3, padding='same', activation='relu')(x)
        x = Conv2D(32, 3, padding='same', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Conv2D(1, 3, padding='same', activation='relu')(x)

        decoder = Model(latent_input, x, name ='decoder')
        output = decoder(encoder(inputs)[2])

        def _vae_loss():
            model_input = K.flatten(inputs)
            model_output = K.flatten(output)

            recon_loss= mse(model_input, model_output)
            # recon_loss *= self.input_shape[0]
            kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

            return K.mean(recon_loss + kl_loss)

        vae = Model(inputs, output, name='vae')
        vae.add_loss(vae_loss())

        return vae, encoder, decoder
    
    
    def _variational_autoencoder(self, orignal_dim):
        
        orignal_dim = int(orignal_dim[0])
        inputs = Input(shape= orignal_dim, name = 'encoder_input')
        x = Dense(64, activation = 'relu', kernel_initializer= 'he_uniform')(inputs)
        x = Dense(128, activation = 'relu', kernel_initializer= 'he_uniform')(x)
        x = Dense(64, activation = 'relu', kernel_initializer= 'he_uniform')(x)
        x = Dense(32, activation = 'relu', kernel_initializer= 'he_uniform')(x)
        x = Dense(30, activation = 'relu', kernel_initializer= 'he_uniform')(x)

        x = Dense(self.latent_dim, activation = 'relu', kernel_initializer= 'he_uniform')(x)
        z_mean = Dense(self.latent_dim, activation = None, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, activation = None, name='z_log_var'
                          ,kernel_initializer = initializers.glorot_normal())(x)
        z = Lambda(self._sampling, output_shape = (self.latent_dim,), name ='z')([z_mean, z_log_var])

        encoder = Model(inputs, [z_mean, z_log_var, z], name ='encoder')

        latent_input = Input(shape=(self.latent_dim,), name = 'z_sampling')
        x = Dense(30, activation = 'relu', kernel_initializer= 'he_uniform')(latent_input)
        x = Dense(32, activation = 'relu', kernel_initializer= 'he_uniform')(x)
        x = Dense(64, activation = 'relu', kernel_initializer= 'he_uniform')(x)
        x = Dense(128, activation = 'relu', kernel_initializer= 'he_uniform')(x)
        x = Dense(64, activation = 'relu', kernel_initializer= 'he_uniform')(x)
        x = Dense(orignal_dim, activation = None, kernel_initializer= None)(x)
        
        decoder = Model(latent_input, x, name ='decoder')
        output = decoder(encoder(inputs)[2])

        def vae_loss():
            model_input = K.flatten(inputs)
            model_output = K.flatten(output)

            recon_loss= mse(model_input, model_output)
            # recon_loss = binary_crossentropy(model_input, model_output)
            # recon_loss *= self.input_shape[0]
            kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

            return K.mean(recon_loss + kl_loss)

        vae = Model(inputs, output, name='vae')
        vae.add_loss(vae_loss())

        return vae, encoder, decoder
    
    
    def fit(self, x_data, y_data, minor_class):
        self.x_data = x_data.values
        self.y_data = y_data.values
        
        input_shape = x_data.shape[1:]
        
        x_minor = x_data[y_data == minor_class]
        y_minor = y_data[y_data == minor_class]
        x_train, x_test, y_train, y_test = train_test_split(x_minor, y_minor, test_size=0.2, random_state=42)
        
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', verbose=1, patience=30, mode='min', restore_best_weights=True)
        
        self.model, self.encoder, self.decoder = self._variational_autoencoder(input_shape)
        
        # adam = Adam(0.0001)
        adadelta = tf.keras.optimizers.Adadelta(learning_rate=1., rho=0.95, epsilon=1e-07, name='Adadelta')
        
        self.model.compile(optimizer=adadelta, loss=None)
        self.model.fit(x= x_train, y= None,
                       validation_data=(x_test, None),
                       callbacks = [early_stopping], 
                       epochs=1000, batch_size=128, verbose=1)
    
#     def resample(self, x_data, y_data, minor_class):
#         self.
        
#         decoder.predict(np.random.normal(0, 1., size=(1, self.latent_dim)))