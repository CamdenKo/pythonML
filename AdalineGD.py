import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

class AdalineGD(object) :
  """
  eta: float
    Learning rate 0.0-1.0
  n_iter: int
    number of iterations

  w_ : 1d array
    WEights after fitting
  errors_: list
    number of misclassifications
  """

  def __init__(self, eta=0.01, n_iter=50):
    self.eta = eta
    self.n_iter = n_iter

  def fit(self, X, y):
    """
    X: array like, shape = [n_samples, n_features]
      training vectors
    y: array-like, shape=[n_samples]

    returns
    self: object
    """
    self.w_ = np.zeros(1 + X.shape[1])
    self.cost_ = []

    for _ in range(self.n_iter):
      output = self.net_input(X)
      errors = (y - output)
      self.w_[1:] += self.eta * X.T.dot(errors)
      self.w_[0] += self.eta * errors.sum()
      cost = (errors**2).sum() / 2.0
      self.cost_.append(cost)

    return self

  def net_input(self, X):
    return np.dot(X, self.w_[1:]) + self.w_[0]

  def activation(self, X):
    return self.net_input(X)

  def predict(self, X):
    return np.where(self.activation(X) >= 0.0, 1, -1)
