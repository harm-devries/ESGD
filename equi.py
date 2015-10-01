import numpy
import theano
from theano import tensor as T
from theano.sandbox import rng_mrg

class ESGD(object):
    """Equilibrated SGD with bias correction of Adam.
    
    
    Parameters
    ----------
    parameters : list
        List of parameters of the model. Must be theano shared variables.
    gradients : list
        List of the gradients w.r.t. each parameter.
    """
    
    def __init__(self, parameters, gradients):
        self.parameters = parameters
        self.gradients = gradients
        
        self.ema_grad = [theano.shared(numpy.zeros_like(p.get_value()))
            for p in self.parameters] # exponential moving average of gradient
        self.ema_precond = [theano.shared(numpy.zeros_like(p.get_value()))
            for p in self.parameters] # exponential moving average of equilibration matrix (preconditioner)
            
        self.t1 = theano.shared(numpy.asarray(0, "float32"))
        self.t2 = theano.shared(numpy.asarray(0, "float32"))
        self.rng = rng_mrg.MRG_RandomStreams(numpy.random.randint(2**30))
    
    def updates(self, learning_rate, beta1, beta2, epsilon):
        """ Returns two updates. A slow one that updates the equilibration preconditioner. It should be called once every 
        X updates. The second update is faster because it uses the saved estimate of the equilibration preconditioner. 
        """
        # Update gradient estimation
        grad = [beta1*old_g + (1-beta1)*g  for old_g, g  in
            zip(self.ema_grad, self.gradients)]
        new_t1 = self.t1 + 1
        print epsilon
        # Update preconditioner
        samples = [self.rng.normal(size=p.shape, avg=0, std=1,
            dtype=theano.config.floatX) for p in self.parameters]
        product = theano.gradient.Lop(self.gradients, self.parameters, samples)
        precond = [beta2*old_precond + (1-beta2)*(p**2)  for old_precond, p in
            zip(self.ema_precond, product)]
        new_t2 = self.t2 + 1

        slow_updates = zip(self.ema_precond, precond)
        slow_updates.append((self.t2, new_t2))
        slow_updates += zip(self.ema_grad, grad)
        slow_updates.append((self.t1, new_t1))
        
        for param, g, precon in zip(self.parameters, self.gradients, precond):
            g_bias_corrected = g/(1-beta1**new_t1)
            precon_bias_corrected = precon/(1-beta2**new_t2)
            update = -learning_rate * g_bias_corrected / (T.sqrt(precon_bias_corrected) + epsilon)
            slow_updates.append((param, param + update))

        fast_updates = zip(self.ema_grad, grad)   
        fast_updates.append((self.t1, new_t1))
        for param, g, precon in zip(self.parameters, grad, self.ema_precond):
            g_bias_corrected = g/(1-beta1**new_t1)
            precon_bias_corrected = precon/(1-beta2**new_t2)
            update = - learning_rate * g_bias_corrected / (T.sqrt(precon_bias_corrected) + epsilon)
            fast_updates.append((param, param + update))
        
        return slow_updates, fast_updates
        
        
class Adam(object):
    """Adam
    
    
    Parameters
    ----------
    parameters : list
        List of parameters of the model. Must be theano shared variables.
    gradients : list
        List of the gradients w.r.t. each parameter.
    """
    
    def __init__(self, parameters, gradients):
        self.parameters = parameters
        self.gradients = gradients
        
        self.ema_grad = [theano.shared(numpy.zeros_like(p.get_value()))
            for p in self.parameters] # exponential moving average of gradient
        self.ema_precond = [theano.shared(numpy.zeros_like(p.get_value()))
            for p in self.parameters] # exponential moving average of equilibration matrix (preconditioner)
            
        self.t1 = theano.shared(numpy.asarray(0, "float32"))
        self.t2 = theano.shared(numpy.asarray(0, "float32"))
        self.rng = rng_mrg.MRG_RandomStreams(numpy.random.randint(2**30))
    
    def updates(self, learning_rate, beta1, beta2, epsilon):
        """ Returns two updates. A slow one that updates the equilibration preconditioner. It should be called once every 
        X updates. The second update is faster because it uses the saved estimate of the equilibration preconditioner. 
        """
        # Update gradient estimation
        grad = [beta1*old_g + (1-beta1)*g  for old_g, g  in
            zip(self.ema_grad, self.gradients)]
        new_t1 = self.t1 + 1
        
        # Update preconditioner
        precond = [beta2*old_precond + (1-beta2)*(p**2)  for old_precond, p in
            zip(self.ema_precond, self.gradients)]
        new_t2 = self.t2 + 1

        slow_updates = zip(self.ema_precond, precond)
        slow_updates.append((self.t2, new_t2))
        slow_updates += zip(self.ema_grad, grad)
        slow_updates.append((self.t1, new_t1))
        
        for param, g, precon in zip(self.parameters, self.gradients, precond):
            g_bias_corrected = g/(1-beta1**new_t1)
            precon_bias_corrected = precon/(1-beta2**new_t2)
            update = -learning_rate * g_bias_corrected / (T.sqrt(precon_bias_corrected) + epsilon)
            slow_updates.append((param, param + update))

        fast_updates = zip(self.ema_grad, grad)   
        fast_updates.append((self.t1, new_t1))
        for param, g, precon in zip(self.parameters, grad, self.ema_precond):
            g_bias_corrected = g/(1-beta1**new_t1)
            precon_bias_corrected = precon/(1-beta2**new_t2)
            update = - learning_rate * g_bias_corrected / (T.sqrt(precon_bias_corrected) + epsilon)
            fast_updates.append((param, param + update))
        
        return slow_updates, fast_updates
