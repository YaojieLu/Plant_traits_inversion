
from scipy import optimize
import numpy as np
import theano
import theano.tensor as tt

para = 3
def func(x, a, b):
    value = x + para * np.exp(x) - a * b**2
    return value

def jac(x, a, b):
    jac = 1 + para * np.exp(x)
    return jac

def x_from_ab(a, b):
    value = np.zeros(len(a))
    for i in range(len(a)):
        value[i] = optimize.newton(func, 1, fprime = jac, args=(a[i], b[i], ))
    return value

class Xf(tt.Op):
    itypes = [tt.dvector, tt.dvector]
    otypes = [tt.dvector]

    def perform(self, node, inputs, outputs):
        a, b = inputs
        x = x_from_ab(a, b)
        outputs[0][0] = np.array(x)

    def grad(self, inputs, g):
        a, b = inputs
        x = self(a, b)
        return [-g[0][0] * (-b**2)/(1 + para * tt.exp(x)), -g[0][0] * (-2*a*b)/(1 + para * tt.exp(x))]

# Evaluate
# perform
att = tt.dvector('att')
btt = tt.dvector('btt')
#f = theano.function([att, btt], Xf()(att, btt))
v1 = np.array([3, 4])
v2 = np.array([5, 6])
#f(v1, v2)
# grad
expr = Xf()(att, btt)
ga = tt.grad(expr[0], att)
gb = tt.grad(expr[0], btt)
print(ga.eval({att:v1, btt:v2}))
print(gb.eval({att:v1, btt:v2}))
