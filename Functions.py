
from scipy import optimize, special
import numpy as np
import theano
import theano.tensor as tt



def func(mu, theta, a):
    thetamu = theta * mu
    value = np.log(mu) + np.logaddexp(0, a*thetamu)
    return value

def jac(mu, theta, a):
    thetamu = theta * mu
    jac = a*theta * special.expit(a*thetamu) + 1 / mu
    return jac

def mu_from_theta(theta):
    return optimize.newton(func, 1, fprime=jac, args=(theta, a,))

class MuFromTheta(tt.Op):
    itypes = [tt.dscalar]
    otypes = [tt.dscalar]

    def perform(self, node, inputs, outputs):
        theta, = inputs
        mu = mu_from_theta(theta)
        outputs[0][0] = np.array(mu)

    def grad(self, inputs, g):
        theta, = inputs
        mu = self(theta)
        thetamu = theta * mu
        return [- g[0] * mu ** 2 / (1 + thetamu + tt.exp(-thetamu))]

x = theano.tensor.dscalar()
f = theano.function([x], MuFromTheta()(x))
out = f(4.0)
print(out)
