import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.data.experimental import AutoShardPolicy
from tensorflow.keras import backend as K, layers, mixed_precision
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Dense, Flatten, \
    Input, Reshape, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

# Local application imports
from py_ml_tools.dataset import get_ifo_data_generator, get_ifo_data, O3
from py_ml_tools.model import ConvLayer, DenseLayer, DropLayer, ModelBuilder, \
    PoolLayer, randomizeLayer, negative_loglikelihood
from py_ml_tools.setup import find_available_GPUs, load_label_datasets, \
    setup_cuda

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
if __name__ == "__main__":
    
    gpus = find_available_GPUs(10000, 1)
    strategy = setup_cuda(gpus, 8000, verbose = True)
    
    #policy = mixed_precision.Policy('mixed_float16')
    #mixed_precision.set_global_policy(policy)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    num_examples_per_batch = 32
    sample_rate_hertz = 8192.0
    onsource_duration_seconds = 1.0
    
    # Create TensorFlow dataset from the generator
    injection_configs = [
        {
            "type" : "cbc",
            "snr"  : \
                {"min_value" : 30, "max_value": 30, "mean_value": 0.5, "std": 20, "distribution_type": "uniform"},
            "injection_chance" : 1.0,
            "padding_seconds" : {"front" : 0.9, "back" : 0.1},
            "args" : {
                "approximant_enum" : \
                    {"value" : 1, "distribution_type": "constant", "dtype" : int}, 
                "mass_1_msun" : \
                    {"min_value" : 5, "max_value": 95, "distribution_type": "uniform"},
                "mass_2_msun" : \
                    {"min_value" : 5, "max_value": 95, "distribution_type": "uniform"},
                "sample_rate_hertz" : \
                    {"value" : sample_rate_hertz, "distribution_type": "constant"},
                "duration_seconds" : \
                    {"value" : onsource_duration_seconds, "distribution_type": "constant"},
                "inclination_radians" : \
                    {"min_value" : 0, "max_value": np.pi, "distribution_type": "uniform"},
                "distance_mpc" : \
                    {"min_value" : 10, "max_value": 1000, "distribution_type": "uniform"},
                "reference_orbital_phase_in" : \
                    {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                "ascending_node_longitude" : \
                    {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                "eccentricity" : \
                    {"min_value" : 0, "max_value": 0.1, "distribution_type": "uniform"},
                "mean_periastron_anomaly" : \
                    {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                "spin_1_in" : \
                    {"min_value" : -0.5, "max_value": 0.5, "distribution_type": "uniform", "num_values" : 3},
                "spin_2_in" : \
                    {"min_value" : -0.5, "max_value": 0.5, "distribution_type": "uniform", "num_values" : 3}
            }
        }
    ]
    
    with strategy.scope():
        
        latent_dim = 2

        # Define the input shape
        encoder_inputs = keras.Input(shape=(8196, 1))

        x = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(2049 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((2049, 64))(x)
        x = layers.Conv1DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv1DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()
        
        # Setting options for data distribution
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA      
        
        def transform_features_labels(features_in, labels_in):
            
            features = {"injections_i" : features_in["injections"][0] }
            labels = {"injections_o" : features_in["injections"][0]  }
            return features, labels
        
        # Creating the noise dataset
        cbc_ds = get_ifo_data_generator(
            time_interval = O3,
            data_labels = ["noise", "glitches", "events"],
            ifo = 'L1',
            injection_configs = injection_configs,
            sample_rate_hertz = sample_rate_hertz,
            onsource_duration_seconds = onsource_duration_seconds,
            max_segment_size = 3600,
            num_examples_per_batch = num_examples_per_batch,
            order = "random",
            seed = 123,
            apply_whitening = True,
            input_keys = ["injections"], 
            output_keys = ["injections"]
        ).with_options(options).map(transform_features_labels).take(int(1.0E6//32))

        vae = VAE(encoder, decoder)
        vae.compile(optimizer=keras.optimizers.Adam())
        vae.fit(cbc_ds, epochs=30)

    import matplotlib.pyplot as plt

    def plot_latent_space(vae, n=30, figsize=15):
        # display an n*n 2D manifold of digits
        digit_size = 28
        scale = 1.0
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = vae.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = digit

        plt.figure(figsize=(figsize, figsize))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="Greys_r")
        plt.savefig("Mnist_test.png")


    plot_latent_space(vae)

