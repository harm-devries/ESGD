#!/usr/bin/env python
# encoding: utf-8
"""
MNIST example of ESGD: Equilibrated Stochastic Gradient Descent
by Harm de Vries and Yann Dauphin
"""

import sys
import os
import cPickle
import gzip
import time

import numpy
from theano.sandbox import rng_mrg

import theano
from theano import tensor as T

theano.config.floatX = 'float32'

def main(n_inputs=784,
         n_hiddens0=1000,
         n_hiddens1=1000,
         n_out=10,
         learning_rate=0.01,
         beta1=0.9,
         beta2=0.999,
         epsilon=10**-6,
         n_updates=250*100,
         batch_size=200,
         restart=0,
         state=None,
         channel=None,
         **kwargs):
    numpy.random.seed(0xeffe)

    print locals()

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = cPickle.load(gzip.open("/data/lisatmp/dauphiya/ddbm/mnist.pkl.gz", 'rb'))

    inds = range(train_x.shape[0])
    numpy.random.shuffle(inds)
    train_x = numpy.ascontiguousarray(train_x[inds])
    train_y = train_y[inds]

    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y.astype('int32'))

    s_valid_x = theano.shared(valid_x)
    s_valid_y = theano.shared(valid_y.astype('int32'))
    
    def init_param(nx, ny, name=None):
        W = numpy.random.uniform(low=-1e-3, high=1e-3, size=(nx,ny)).astype('float32')
        return theano.shared(W)

    W0 = init_param(n_inputs, n_hiddens0)
    W1 = init_param(n_hiddens0, n_hiddens1)
    W2 = init_param(n_hiddens1, n_out)
    b0 = theano.shared(numpy.zeros(n_hiddens0, 'float32'))
    b1 = theano.shared(numpy.zeros(n_hiddens1, 'float32'))
    b2 = theano.shared(numpy.zeros(n_out, 'float32'))
    params = [W0, b0, W1, b1, W2, b2]

    input = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar('i')
    vec = T.vector('v')
    
    def rect(x):
        return T.switch(x > 0, x, 0)
        
    hidden = rect(T.dot(input, W0) + b0)
    hidden = rect(T.dot(hidden, W1) + b1)
    hidden = T.dot(hidden, W2) + b2

    #e_x = T.exp(hidden - theano.gradient.zero_grad(hidden.max(axis=1)[:, None]))
    #output = e_x / e_x.sum(axis=1)[:, None]
    output = T.nnet.softmax(hidden)
    
    def cost(x, y):
        ind = T.arange(y.shape[0])*x.shape[1] + y
        flat_x = T.reshape(x, (x.shape[0]*x.shape[1],))
        return -T.log(flat_x[ind]) 
        
    loss = cost(output, y).mean()
    misclass = T.neq(y, output.argmax(axis=1)).mean()
    gparams = T.grad(loss, params)

    givens = { input : s_train_x[index * batch_size:(index + 1) * batch_size], 
               y : s_train_y[index * batch_size:(index + 1) * batch_size]}
    givens_full = {input : s_train_x, y: s_train_y}
    givens_valid = {input: s_valid_x, y: s_valid_y}
        
    n_batches = len(train_x) / batch_size

    from equi import ESGD, Adam
    
    opt = ESGD(params, gparams)
    
    slow_updates, fast_updates = opt.updates(learning_rate, beta1, beta2, epsilon)
    
    f_train = theano.function([index], loss, givens=givens, updates=slow_updates)
    f_train_fast = theano.function([index], loss, givens=givens, updates=fast_updates)
    f_loss = theano.function([], [loss, misclass], givens=givens_full)
    f_val_loss = theano.function([],[loss, misclass], givens=givens_valid)

    begin = time.time()
    best_train_error = float("inf")

    print "Training..."
    for update in range(n_updates):
        if (update % 1 == 0):
            f_train(update % n_batches)
        else:
            f_train_fast(update % n_batches)

        if update % (n_batches) == 0:
            loss, misclass = f_loss()
            val_loss, val_misclass = f_val_loss()
            print  "[%d, %f, %f, %f, %f, %f]," % (update / n_batches, time.time() - begin, loss, misclass, val_loss, val_misclass)
            
if __name__ == "__main__":
    main()
