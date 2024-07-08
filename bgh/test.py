import tensorflow as tf

ds = tf.contrib.distributions
mix = 0.3
bimix_gauss = ds.Mixture(
  cat=ds.Categorical(probs=[mix, 1.-mix]),
  components=[
    ds.Normal(loc=-1., scale=0.1),
    ds.Normal(loc=+1., scale=0.5),
])

# Plot the PDF.
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
x = tf.linspace(-2., 3., int(1e4)).eval()
plt.plot(x, bimix_gauss.prob(x).eval());