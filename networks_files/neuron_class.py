#class Neuron:
  #  weights = []
  #   activation_func = "sigmoid"  # default
  #
  #   def __init__(self, weights, activation_func):
  #       self.weights = weights
  #       self.activation_func = activation_func
  #   def output(self, input, activation_func):
  #       weighted_sum = np.dot(input, self.weights)
  #       self.activation_func = activation_func
  #       if self.activation_func == "sigmoid":
  #           return sigmoid(weighted_sum)
