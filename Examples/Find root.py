
from scipy import optimize
import numpy as np
import theano
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
    
    def grad(self, inputs, output_gradients):
        a, b = inputs
        x = self(a, b)
        g, = output_gradients
        return [-g * (-b**2)/(1 + env * tt.exp(x)), -g * (-2*a*b)/(1 + env * tt.exp(x))]

att = tt.dscalar('att')
btt = tt.dscalar('btt')
f = theano.function([att, btt], Xf()(att, btt))
theano.printing.pydotprint(f, outfile="model.png", var_with_name_simple=True)
expr = Xf()(att, btt)

for i in range(len(envod)):
    env = envod[i]
    ga = tt.grad(expr, att)
    print(f(3, 4))
    print(ga.eval({att:3, btt:4}))