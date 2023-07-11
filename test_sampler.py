import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

n_components = 20
latent_dim = 10
batch_size = 10

# Test data
z_mean = tf.random.normal([batch_size, n_components, latent_dim])
z_log_var = tf.random.normal([batch_size, n_components, latent_dim])
z_weight = tf.random.normal([batch_size, n_components, latent_dim])

# Definition of the sampling function (as provided in the original question)
def sampling(args):
    z_mean, z_log_var, z_weight = args
    z_weight = tf.nn.softmax(tf.reshape(z_weight, [None, n_components, latent_dim]), axis = -1)  
    z_mean = tf.reshape(z_mean, [None, n_components, latent_dim])
    z_log_var = tf.math.softplus(tf.reshape(z_log_var, [None, n_components, latent_dim]))  

    all_samples = []
    for i in range(latent_dim):
        mixture_distribution = tfd.Categorical(probs=z_weight[..., i])
        components_distribution = tfd.Normal(loc=z_mean[..., i], scale=z_log_var[..., i])
        distribution = tfd.MixtureSameFamily(mixture_distribution=mixture_distribution,
                                             components_distribution=components_distribution)
        all_samples.append(distribution.sample())

    return tf.stack(all_samples, axis=-1)

# Call the sampling function and print the result
result = sampling((z_mean, z_log_var, z_weight))
print(result)