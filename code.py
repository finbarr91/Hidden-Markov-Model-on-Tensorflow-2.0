# Hidden-Markov-Model-on-Tensorflow-2.0
Written in Python
import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions # Making a shortcut fro later on
initial_distribution = tfd.Categorical(probs=[0.8, 0.2]) # refer to the point 2 above
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], # refer to the points 3 and 4 above
                                                 [0.2, 0.8]])
obseravtion_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.]) # refer to point 5 above

# the loc argument represents the mean and the scale is the standard deviation
model = tfd.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution=  obseravtion_distribution,
    num_steps = 7)

mean = model.mean()
# due to the way tensorFlow works on a lower level, we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorFlow we need to use tf.compat.v1. Session() rather than just tf.Session()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())
