
from scipy import optimize
import numpy as np
import theano.tensor as tt

envod = np.array([1, 2])

def func(x, a, b, env):
    value = x + env * np.exp(x) - a * b**2
    return value

def x_from_ab(a, b, env):
    value = optimize.newton(func, 4, args = (a, b, env))
    return value

class Xf(tt.Op):
    itypes = [tt.dscalar, tt.dscalar]
    otypes = [tt.dscalar]

    def perform(self, node, inputs, outputs):
        a, b = inputs
        x = x_from_ab(a, b, env)
        outputs[0][0] = np.array(x)
    
    def grad(self, inputs, g):
        a, b = inputs
        x = self(a, b)
        dx = 1 + env * tt.exp(x)
        da = -b**2
        db = -2 * a * b
        return [-g[0] * da / dx, -g[0] * db / dx]

for i in range(len(envod)):
    env = envod[i]
    tt.verify_grad(Xf(), [3.0, 4.0], rng=np.random)