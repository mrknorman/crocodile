# Third party imports
import numpy as np
import tensorflow as tf
from bokeh.io import output_file
from bokeh.models import Span, ColumnDataSource
from bokeh.plotting import figure, save, output_file
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

tfd = tfp.distributions

# Number of mixture components
n_components = 5

def sampling(
        z_params: tuple, 
        batch_size: int,
        n_components: int, 
        latent_dim: int
    ) -> tf.Tensor:
    """
    Function to sample from a Gaussian mixture model in the latent space.

    Parameters
    ----------
    z_params : tuple
        Tuple containing z_mean, z_log_var and z_weight tensors
    batch_size : int
        Batch size for reshaping tensors
    n_components : int
        Number of Gaussian distributions to mix
    latent_dim : int
        Dimensionality of the latent space

    Returns
    -------
    tf.Tensor
        Sampled tensor from the Gaussian mixture model
    """
    z_mean, z_log_var, z_weight = z_params

    # Convert to proper shapes and apply softmax to weights
    z_weight = tf.nn.softmax(tf.reshape(z_weight, [batch_size, latent_dim, n_components]), axis=-1)
    z_mean = tf.reshape(z_mean, [batch_size, latent_dim, n_components])
    z_log_var = tf.math.softplus(tf.reshape(z_log_var, [batch_size, latent_dim, n_components]))

    # Create mixture and components distributions
    mixture_distribution = tfd.Categorical(probs=z_weight)
    components_distribution = tfd.Normal(loc=z_mean, scale=z_log_var)

    # Sample from the mixed distributions
    distribution = tfd.MixtureSameFamily(mixture_distribution=mixture_distribution,
                                         components_distribution=components_distribution)

    return distribution.sample()

def create_vae(
        input_shape: tuple, 
        latent_dim: int, 
        n_components: int, 
        kernel_size: int = 8, 
        filters: int = 16, 
        batch_size: int = 32
    ) -> Model:
    """
    Function to create a Variational Autoencoder (VAE) with a Gaussian mixture model in the latent space.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input data
    latent_dim : int
        Dimensionality of the latent space
    n_components : int
        Number of Gaussian distributions to mix
    kernel_size : int, optional
        Size of the convolution kernel, by default 3
    filters : int, optional
        Number of filters for the convolution layers, by default 16
    batch_size : int, optional
        Size of the batches for training, by default 32

    Returns
    -------
    Model
        A Variational Autoencoder model
    """
    # Build Encoder
    inputs = Input(shape=input_shape, name='injections_i')
    x = inputs
    for _ in range(2):
        #filters *= 2
        x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',  padding='same')(x)
        x = MaxPooling1D(pool_size=kernel_size-1)(x)
    
    shape = K.int_shape(x)
    x = Flatten()(x)
    z_mean = Dense(latent_dim * n_components, name='z_mean')(x)
    z_log_var = Dense(latent_dim * n_components, name='z_log_var')(x)
    z_weight = Dense(latent_dim * n_components, name='z_weight')(x)
    z = sampling((z_mean, z_log_var, z_weight), batch_size, n_components, latent_dim)
    
    encoder = Model(inputs, [z_mean, z_log_var, z_weight, z], name='encoder')
    
    # Build Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2]))(x)

    for _ in range(2):
        x = Conv1DTranspose(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = Conv1DTranspose(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = UpSampling1D(size=kernel_size-1)(x)
        #filters //= 2

    outputs = Conv1DTranspose(filters=1, kernel_size=kernel_size, activation='linear', padding='same', name='decoder_output')(x)
    decoder = Model(latent_inputs, outputs, name='injections_o')

    # Build VAE model
    outputs = decoder(encoder(inputs)[3])
    vae = Model(inputs, outputs, name='vae')

    # Define VAE loss
    reconstruction_loss = K.mean(K.square(K.cast(inputs, tf.float16) - outputs))
    #nll_loss = -K.sum(tfd.Normal(loc=z_mean, scale=z_log_var).log_prob(input_conditions), axis=-1)
    vae_loss = K.mean(reconstruction_loss)  # Here you can also add KL divergence loss if required
    vae.add_loss(vae_loss)
    
    return vae

def plot_gravitational_wave(signal, sample_rate):
    # Generate time array
    time = np.arange(0, len(signal) / sample_rate, 1 / sample_rate)
    
    # Output to file
    output_file("test_waveform.html")
    
    # Create a new plot with a title and axis labels
    p = figure(title="Gravitational wave signal", x_axis_label='Time (s)', y_axis_label='Amplitude')
    
    # Add a line renderer with legend and line thickness
    p.line(time, signal, legend_label="Gravitational Wave", line_width=2)
    
    # Show the results
    save(p)
    
# Then add the plotting function
def plot_input_output_comparison(original_data, reconstructed_data, sample_rate):
    # Generate time array
    time = np.arange(0, len(original_data) / sample_rate, 1 / sample_rate)
    
    # Output to file
    output_file("vae_input_output_comparison.html")
    
    # Create a new plot with a title and axis labels
    p = figure(title="Input vs Output", x_axis_label='Time (s)', y_axis_label='Amplitude')
    
    # Add a line renderer for the original data and the reconstructed data
    p.line(time, original_data, legend_label="Original Data", line_width=2, color='blue')
    p.line(time, reconstructed_data, legend_label="Reconstructed Data", line_width=2, color='red')
    
    # Show the results
    save(p)

if __name__ == "__main__":
    
    gpus = find_available_GPUs(10000, 1)
    strategy = setup_cuda(gpus, 8000, verbose = True)
    
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
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
                {"min_value" : 0.5, "max_value": 100, "mean_value": 0.5, "std": 20, "distribution_type": "uniform"},
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
        
        #plot_gravitational_wave(next(cbc_ds)[0]["injections"][0][0].numpy(), sample_rate_hertz)
        
        input_shape = (int(sample_rate_hertz*onsource_duration_seconds), 1)  # Modify this based on your input shape
        latent_dim = 20  # Number
        
        model = create_vae(
            input_shape, 
            latent_dim, 
            n_components = 2, 
            kernel_size=3, 
            filters=64
        )
        
        # Compile and train
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer)
        
        model.summary()
        
        model.fit(cbc_ds)
        model.save_weights('model_weights.h5')
        
        # Let's say you are using the first batch of your dataset for testing
        example_batch = next(iter(cbc_ds))
        example_input = example_batch[0]['injections_i']
        example_output = model.predict(example_input)
        
        # Reshape your tensors to 1D arrays before plotting
        example_input_1D = tf.reshape(example_input[0], [-1]).numpy()
        example_output_1D = tf.reshape(example_output[0], [-1]).numpy()

        # Now use the plot function with the original data (example_input_1D) and the reconstructed data (example_output_1D)
        plot_input_output_comparison(example_input_1D, example_output_1D, sample_rate_hertz)