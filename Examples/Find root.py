
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
    return optimize.newton(func, 1, fprime = jac, args=(a, b))

class Xf(tt.Op):
    itypes = [tt.dscalar, tt.dscalar]
    otypes = [tt.dscalar]

    def perform(self, node, inputs, outputs):
        a, b = inputs
        x = x_from_ab(a, b)
        outputs[0][0] = np.array(x)

    def grad(self, inputs, g):
        a, b = inputs
        x = self(a, b)
        return [-g[0] * (-b**2)/(1 + para * tt.exp(x)), -g[0] * (-2*a*b)/(1 + para * tt.exp(x))]

att = tt.dscalar('att')
btt = tt.dscalar('btt')
f = theano.function([att, btt], Xf()(att, btt))
expr = Xf()(att, btt)
ga = tt.grad(expr, att)
gb = tt.grad(expr, btt)
print(ga.eval({att:3, btt:3}))
print(gb.eval({att:3, btt:3}))
