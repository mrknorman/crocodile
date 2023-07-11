import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.data.experimental import AutoShardPolicy
from tensorflow.keras import backend as K, layers, mixed_precision
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Dense, Flatten, \
    Input, Reshape, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

import numpy as np

from bokeh.io import output_file
from bokeh.models import Span, ColumnDataSource
from bokeh.plotting import figure, save, output_file

# Local application imports
from py_ml_tools.dataset import get_ifo_data_generator, O3
from py_ml_tools.setup import find_available_GPUs, setup_cuda

from tensorflow import keras

# Define the residual block
def residual_block(x, filters, dilation_rate):
    conv = layers.Conv1D(filters, 3, padding="same", dilation_rate=dilation_rate)(x)
    conv = layers.ReLU()(conv)

    conv = layers.Conv1D(filters, 3, padding="same", dilation_rate=dilation_rate)(conv)
    out = layers.ReLU()(conv)
    
    x = layers.Conv1D(filters, 1, padding="same", dilation_rate=dilation_rate)(x)

    # Skip connection
    out = x + out
    return out

# Define the transposed residual block
def transposed_residual_block(x, filters, dilation_rate):
    conv = layers.Conv1DTranspose(filters, 3, padding="same", dilation_rate=dilation_rate)(x)
    conv = layers.ReLU()(conv)

    conv = layers.Conv1DTranspose(filters, 3, padding="same", dilation_rate=dilation_rate)(conv)
    out = layers.ReLU()(conv)
        
    x = layers.Conv1DTranspose(filters, 1, padding="same", dilation_rate=dilation_rate)(x)

    # Skip connection
    out = x + out
    return out

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


"""
 x = layers.Conv1D(64, 128, activation="relu", padding = "same")(encoder_inputs)
    x = layers.MaxPool1D(4)(x)
    x = layers.Conv1D(32, 64, activation="relu", padding = "same")(x)
    x = layers.Conv1D(32, 64, activation="relu", padding = "same")(x)
    x = layers.MaxPool1D(4)(x)
    x = layers.Conv1D(16, 64, activation="relu", padding = "same")(x)
    x = layers.Conv1D(16, 64, activation="relu", padding = "same")(x)
    x = layers.Conv1D(16, 64, activation="relu", padding = "same")(x)
    
    x = layers.Conv1DTranspose(16, 64, activation="relu", padding = "same")(x)
    x = layers.Conv1DTranspose(16, 64, activation="relu", padding = "same")(x)
    x = layers.Conv1DTranspose(16, 64, activation="relu", padding = "same")(x)
    x = layers.UpSampling1D(4)(x)
    x = layers.Conv1DTranspose(32, 64, activation="relu", padding = "same")(x)
    x = layers.Conv1DTranspose(32, 64, activation="relu", padding = "same")(x)
    x = layers.UpSampling1D(4)(x)
"""

def create_vae(input_shape, latent_dim):
    # Encoder
    encoder_inputs = layers.Input(shape=input_shape, name = 'injections_i')
    
    x = layers.Conv1D(64, 7, padding="same")(encoder_inputs)  # Initial Convolution before residual blocks
    x = layers.ReLU()(x)

    x = residual_block(x, 64, 2)
    x = layers.MaxPool1D(2)(x)

    x = residual_block(x, 64, 2)
    x = layers.MaxPool1D(2)(x)
    
    x = residual_block(x, 64, 2)
    x = layers.MaxPool1D(2)(x)
    
    x = residual_block(x, 64, 2)
    x = layers.MaxPool1D(2)(x)
    
    x = residual_block(x, 64, 2)
    x = layers.MaxPool1D(2)(x)
    
    x = residual_block(x, 64, 2)
    x = layers.MaxPool1D(2)(x)
    
    x = residual_block(x, 64, 2)
    x = layers.MaxPool1D(2)(x)

    x = residual_block(x, 128, 2)
    
    x = layers.Flatten()(x)
    x = layers.Dense(64,activation="relu")(x)
        
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    
    x = layers.Dense(64, activation="relu")(decoder_inputs)
    x = layers.Reshape((64, 1))(x)
    
    x = transposed_residual_block(x, 128, 2)
    
    x = layers.UpSampling1D(2)(x)
    x = transposed_residual_block(x, 64, 2)
    
    x = layers.UpSampling1D(2)(x)
    x = transposed_residual_block(x, 64, 2)
    
    x = layers.UpSampling1D(2)(x)
    x = transposed_residual_block(x, 64, 2)
    
    x = layers.UpSampling1D(2)(x)
    x = transposed_residual_block(x, 64, 2)
    
    x = layers.UpSampling1D(2)(x)
    x = transposed_residual_block(x, 64, 2)
    
    x = layers.UpSampling1D(2)(x)
    x = transposed_residual_block(x, 64, 2)
    
    x = layers.UpSampling1D(2)(x)
    x = transposed_residual_block(x, 64, 2)
    
    decoder_outputs = layers.Conv1DTranspose(1, 7, padding='same')(x)
    decoder = Model(decoder_inputs, decoder_outputs, name='injections_o')
    
    # VAE
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = Model(encoder_inputs, outputs, name='vae')

    # Add VAE loss
    reconstruction_loss = tf.reduce_mean(tf.square(encoder_inputs - outputs))
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var + 1.0E-5) + 1) 
    vae_loss = reconstruction_loss #+ kl_loss
    
    # Return separate losses for monitoring
    vae.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
    vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    
    vae.add_loss(vae_loss)

    return vae

def plot_input_output_comparison(original_data, reconstructed_data, sample_rate, separation=10):
    # Output to file
    output_file("vae_input_output_comparison.html")

    # Create a new plot with a title and axis labels
    p = figure(title="Input vs Output", x_axis_label='Time (s)', y_axis_label='Amplitude')
    
    for i in range(len(original_data)):
        # Generate time array for each waveform
        time = np.arange(0, len(original_data[i]) / sample_rate, 1 / sample_rate)
        
        # Add a line renderer for the original data and the reconstructed data
        p.line(time, original_data[i] + i*separation, legend_label=f"Original Data {i+1}", line_width=2, color='blue')
        p.line(time, reconstructed_data[i] + i*separation, legend_label=f"Reconstructed Data {i+1}", line_width=2, color='red')
    
    p.legend.visible=False 
    
    # Show the results
    save(p)

class PlotInputOutputCallback(keras.callbacks.Callback):
    def __init__(self, example_input, sample_rate_hertz, separation=10):
        super(PlotInputOutputCallback, self).__init__()
        self.example_input = example_input
        self.sample_rate_hertz = sample_rate_hertz
        self.separation = separation

    def on_epoch_end(self, epoch, logs=None):
        example_output = self.model.predict(self.example_input)
        # Select the first 10 waveforms
        example_input_1D = [tf.reshape(inp, [-1]).numpy() for inp in self.example_input]
        example_output_1D = [tf.reshape(out, [-1]).numpy() for out in example_output]
        
        plot_input_output_comparison(example_input_1D[:10], example_output_1D[:10], self.sample_rate_hertz, self.separation)

if __name__ == "__main__":
    gpus = find_available_GPUs(10000, 1)
    strategy = setup_cuda(gpus, 8000, verbose=True)

    num_examples_per_batch = 32
    sample_rate_hertz = 8192.0
    onsource_duration_seconds = 1.0
    
     # Create TensorFlow dataset from the generator
    injection_configs = [
        {
            "type" : "cbc",
            "snr"  : \
                {"min_value" : 100, "max_value": 100, "mean_value": 0.5, "std": 20, "distribution_type": "uniform"},
            "injection_chance" : 1.0,
            "padding_seconds" : {"front" : 0.9, "back" : 0.1},
            "args" : {
                "approximant_enum" : \
                    {"value" : 1, "distribution_type": "constant", "dtype" : int}, 
                "mass_1_msun" : \
                    {"min_value" : 5, "max_value": 30, "mean_value": 5, "std": 20, "distribution_type": "normal"},
                "mass_2_msun" : \
                    {"min_value" : 5, "max_value": 30, "mean_value": 5, "std": 20, "distribution_type": "normal"},
                "sample_rate_hertz" : \
                    {"value" : sample_rate_hertz, "distribution_type": "constant"},
                "duration_seconds" : \
                    {"value" : onsource_duration_seconds, "distribution_type": "constant"},
                "inclination_radians" : \
                    {"min_value" : 0, "max_value": np.pi, "distribution_type": "uniform"},
                "distance_mpc" : \
                    {"min_value" : 10, "max_value": 1000, "distribution_type": "uniform"},
                "reference_orbital_phase_in" : \
                    {"min_value" : 0, "max_value": 0, "distribution_type": "uniform"},
                "ascending_node_longitude" : \
                    {"min_value" : 0, "max_value": 0, "distribution_type": "uniform"},
                "eccentricity" : \
                    {"min_value" : 0, "max_value": 0, "distribution_type": "uniform"},
                "mean_periastron_anomaly" : \
                    {"min_value" : 0, "max_value": 0, "distribution_type": "uniform"},
                "spin_1_in" : \
                    {"min_value" : 0, "max_value": 0, "distribution_type": "uniform", "num_values" : 3},
                "spin_2_in" : \
                    {"min_value" : 0, "max_value": 0, "distribution_type": "uniform", "num_values" : 3}
            }
        }
    ]
    
    # Setting options for data distribution
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA      
    
    def transform_features_labels(features_in, labels_in):    
        
        injections = tf.cast(features_in["injections"][0], tf.float32)
                
        features = {"injections_i" : injections  / tf.reduce_mean(tf.math.abs(injections), axis=1, keepdims=True)}
        labels = {"injections_o" : injections  / tf.reduce_mean(tf.math.abs(injections), axis=1, keepdims=True)}
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
    ).with_options(options).map(transform_features_labels)
    
    input_shape = (int(sample_rate_hertz * onsource_duration_seconds), 1)  # Modify this based on your input shape
    latent_dim = 2  # Number

    with strategy.scope():
        
        model = create_vae(input_shape, latent_dim)
        model.summary()
        model.compile(optimizer=Adam())
        
        # Let's say you are using the first batch of your dataset for testing
        example_batch = next(iter(cbc_ds))
        example_input = example_batch[0]['injections_i']
        
        plot_callback = PlotInputOutputCallback(example_input, sample_rate_hertz)

        # train the model
        model.fit(cbc_ds, epochs=10, steps_per_epoch=int(1.0E5//32), callbacks=[plot_callback])

        # Save the model weights
        model.save_weights('model_weights.h5')
        

        """
        example_output = model.predict(example_input)
        
        # Reshape your tensors to 1D arrays before plotting
        example_input_1D = tf.reshape(example_input[0], [-1]).numpy()
        example_output_1D = tf.reshape(example_output[0], [-1]).numpy()
        
        # Now use the plot function with the original data (example_input_1D) and the reconstructed data (example_output_1D)
        plot_input_output_comparison(example_input_1D, example_output_1D, sample_rate_hertz)"""