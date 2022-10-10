from collections import defaultdict
import numpy as np

## HINT: Lab 2 might be helpful...

class BasicOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def apply_gradients(self, weights, grads):
        # TODO: Update the weights using basic stochastic gradient descent
        weights = weights - self.learning_rate * grads
        return weights


class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate

        self.beta = beta
        self.epsilon = epsilon

        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, weights, grads):
        # TODO: Implement RMSProp optimization
        # Refer to the lab on Optimizers for a better understanding!
        self.v[0] = self.beta*self.v[0] + (1-self.beta)*(grads**2)
        weights = weights - self.learning_rate/(np.sqrt(self.v[0])+self.epsilon)*grads
        return weights


class Adam:
    def __init__(
        self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):
        self.amsgrad = amsgrad

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)  # First moment zero vector
        self.v = defaultdict(lambda: 0)  # Second moment zero vector.
        # Expected value of first moment vector
        self.m_hat = defaultdict(lambda: 0)
        # Expected value of second moment vector
        self.v_hat = defaultdict(lambda: 0)
        self.t = 0  # Time counter

    def apply_gradients(self, weights, grads):
        # TODO: Implement Adam optimization
        # Refer to the lab on Optimizers for a better understanding!
        self.t += 1
        self.m[self.t] = self.beta_1*self.m[self.t-1] + (1-self.beta_1)*grads
        self.v[self.t] = self.beta_2*self.v[self.t-1] + (1-self.beta_2)*(grads**2)

        self.m_hat[self.t] = self.m[self.t-1]/(1-np.power(self.beta_1, self.t))
        self.v_hat[self.t] = self.v[self.t-1]/(1-np.power(self.beta_2, self.t))
        weights = weights - self.learning_rate * self.m_hat[self.t]/(np.sqrt(self.v_hat[self.t])+self.epsilon)
        return
