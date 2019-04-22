## This is not working well because somewhere around  
# gradient calculation is off. But it passes test 
# for single layer test in the notebook :/

import numpy as np

from sequential_model.layers import *
#from sequential_model.fast_layers import *
from sequential_model.layer_utils import *

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
  
  a, fc_cache = affine_forward(x, w, b)
  y, bn_cache = batchnorm_forward(a, gamma, beta,bn_param)
  out, relu_cache = relu_forward(y)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_batchnorm_relu_backward(dout, cache):

  fc_cache, bn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dy, dgamma, dbeta = batchnorm_backward(da, bn_cache) 
  dx, dw, db = affine_backward(dy, fc_cache)
  return dx, dw, db, dgamma, dbeta


def conv_batchnorm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):

  a, conv_cache = conv_forward_naive(x, w, b, conv_param)
  norm, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(norm)
  out, pool_cache = max_pool_forward_naive(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache, bn_cache)
  return out, cache

def conv_batchnorm_relu_pool_backward(dout, cache):
 
  conv_cache, relu_cache, pool_cache, bn_cache = cache
  ds = max_pool_backward_naive(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dnorm, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
  dx, dw, db = conv_backward_naive(dnorm, conv_cache)
  return dx, dw, db, dgamma, dbeta

class ConvNet(object):
 
  
  def __init__(self, num_filters, filter_sizes, hidden_dims, conv_param, pool_param,
                input_dim=(1, 128, 100),#(3, 32, 32), 
                num_classes=2, weight_scale=1e-3, reg=0.0,
               use_batchnorm = False, seed= None, 
               dtype=np.float32):
    """
    Initialize a new network with the architecture: 
    [conv-relu-pool]xN-[affine]xM-[Softmax]
    
    JODO: update
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: A list of number of filters to use in the convolutional layers
    - filter_sizes: A list of size of filters to use in the convolutional layers
    - hidden_dims: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - use_batchnorm: if spatial batch normalization is used
    - reg: Scalar giving L2 regularization strength
    - conv_param: 
    - pool_param:
    - dtype: numpy datatype to use for computation.
    """
    assert len(num_filters) == len(filter_sizes), "Filters specs do not align!"
    self.params = {}
    self.use_batchnorm = use_batchnorm
    self.num_CRP_layers = len(num_filters)
    self.num_aff_layers = len(hidden_dims)
    self.reg = reg
    self.dtype = dtype
    self.pool_param = pool_param
    self.conv_param = conv_param

    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'cW1' #
    # and 'cb1'; use keys 'aW1' and 'ab1' for the weights and biases of the       #
    # hidden affine layers.                                              #
    ############################################################################
    cur_input_dim = input_dim
    ## Dimension countdown:
    # Conv:
    # X: (N, C, H, W)
    # W1 (conv): (F, C, HH, WW)
    # b1 (conv): (F,)
    # out: (N, F, H, W) thanks to padding

    # relu:
    # out: same as previous
    # pooling: 
    # out: (N, F, H_prime, W_prime)
    for i in range(self.num_CRP_layers):
        C, H, W = cur_input_dim
        ## Calculate output dimension after conv layer
        pad = conv_param['pad']
        stride = conv_param['stride']
        filter_size = filter_sizes[i]
        H += 2 * pad
        W += 2 * pad
        out_h = int((H - filter_size) / stride + 1)
        out_w = int((W - filter_size) / stride + 1)

        self.params['cW' + str(i)] = np.random.normal(0, weight_scale, (num_filters[i], C, filter_sizes[i], filter_sizes[i]))
        self.params['cb' + str(i)] = np.zeros(num_filters[i])#np.zeros((1,num_filters[i])) 
        
        ## Calculate output dimension after pooling layer
        pooling_stride = self.pool_param['stride']
        pool_h = self.pool_param['pool_height']
        pool_w = self.pool_param['pool_width']

        H_prime = int((out_h-pool_h)/pooling_stride+1)
        W_prime = int((out_w-pool_w)/pooling_stride+1)
        volumn = num_filters[i]
        cur_input_dim = (volumn, H_prime, W_prime)



   
    ##  The affine right after CRP layer:
    # x: (N, F, H_prime, W_prime)
    # W2: (F, H_prime, W_prime, hidden_dim)
    # b2: (hidden_dim,)
    # out: (N, hidden_dim)
    # relu:
    # out: same as previous
    
    C, H_prime, W_prime = cur_input_dim
    F = num_filters[i]
    self.params['aW0'] = np.random.normal(0, weight_scale, (F, H_prime, W_prime, hidden_dims[0]))
    self.params['ab0'] = np.zeros((1,hidden_dims[0]))

    ## The next few affine+relu layers
    for j in range(1,self.num_aff_layers): ## since last affine layer needs weights too
        self.params['aW'+str(j)] = np.random.normal(0, weight_scale, (hidden_dims[j-1], hidden_dims[j]))
        self.params['ab'+str(j)] = np.zeros((1,hidden_dims[j]))
            
    ## Last affine only layer before softmax
    last_aff_layer = self.num_aff_layers
    self.params['aW'+str(last_aff_layer)] = np.random.normal(0, weight_scale, (hidden_dims[last_aff_layer-1], num_classes))
    self.params['ab'+str(last_aff_layer)] = np.zeros((1, num_classes))
    # out: (N, num_classes)

    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_aff_layers + self.num_CRP_layers)]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    for k, v in self.params.items():#iteritems():
      self.params[k] = v.astype(dtype)
  

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    input_data = X
    CRP_cache = {}
    aff_cache ={}
    ## Forward pass
    # conv-relu-pool
    for i in range(self.num_CRP_layers):
        if self.use_batchnorm:

            C = input_data.shape[1]
            gamma = np.ones(C)
            beta = np.zeros(C)
            out_CRP_norm, cache_CRP_norm = conv_batchnorm_relu_pool_forward(input_data,
                                                                self.params['cW'+str(i)], 
                                                                self.params['cb'+str(i)],
                                                                self.conv_param,
                                                                self.pool_param,  
                                                                gamma,beta,self.bn_params[i])
            # JOTO: cache?
            input_data = out_CRP_norm
            CRP_cache[i] = (cache_CRP, cache_CRP_norm)
        else: 
            out_CRP, cache_CRP = conv_relu_pool_forward(input_data, 
                                                self.params['cW'+str(i)], 
                                                self.params['cb'+str(i)],                                      
                                                self.conv_param, self.pool_param)
        
            input_data = out_CRP
            CRP_cache[i] = cache_CRP
        


    for j in range(self.num_aff_layers):
        if self.use_batchnorm: 
            out_dim = self.params['aW' + str(j)].shape[1]
            gamma = np.ones(out_dim) ##JODO: need to initialize for each iteration? 
            beta = np.zeros(out_dim) ## No because these 2 parameters are batch specific? but then how to use it during test?
            out_aff_norm, cache_aff_norm = affine_batchnorm_relu_forward(input_data, 
                                        self.params['aW' + str(j)], 
                                        self.params['ab' + str(j)],
                                         gamma, 
                                         beta, 
                                         self.bn_params[i+j])
            aff_cache[j] = (cache_aff, cache_aff_norm)
            input_data = out_aff_norm
        else:
            out_aff, cache_aff = affine_relu_forward(input_data, 
                                            self.params['aW'+str(j)],
                                            self.params['ab'+str(j)])
            aff_cache[j] = cache_aff
            input_data = out_aff

        
    last_aff_layer =self.num_aff_layers
    out, aff_cache[last_aff_layer] = affine_forward(input_data, self.params['aW'+str(last_aff_layer)],
                                            self.params['ab'+str(last_aff_layer)])
    # last affine-softmax
    
    if y is None:

        #scores = softmax_output(out3)
        scores = out
        return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    
    ## Backprop:
    data_loss, dout = softmax_loss(out, y)
    for i in range(self.num_CRP_layers+self.num_aff_layers+1):
        weight_loss = 0
        if i >= self.num_CRP_layers:
            j = i - self.num_CRP_layers
            w = self.params['aW'+str(j)].copy().flatten()
            w_norm = np.sum(w*w)
        else:
            w = self.params['cW'+str(i)].copy().flatten()
            w_norm = np.sum(w*w)
        weight_loss += w_norm
    
    loss = data_loss + 0.5*self.reg*weight_loss

    ## last affine layer:
    dx, dw, db = affine_backward(dout, aff_cache[last_aff_layer])
    dw += self.reg*self.params['aW' + str(last_aff_layer)]
    grads['aW'+str(last_aff_layer)] = dw
    grads['ab' + str(last_aff_layer)] = db

    ## Backprop affine-relu layer:
    for i in reversed(range(self.num_aff_layers)):
        if self.use_batchnorm:
            
            dx, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dx, aff_cache[i])
            grads['agamma' +str(i)] = dgamma
            grads['abeta' +str(i)] = dbeta
        else:
            dx, dw, db = affine_relu_backward(dx, aff_cache[i])
        
        dw += self.reg*self.params['aW'+str(i)]
        
        grads['aW'+str(i)] = dw
        grads['ab'+str(i)] = db



    ## Backprop conv-relu-pool layer:

    for j in reversed(range(self.num_CRP_layers)):
        
        if self.use_batchnorm:
            dx, dw, db, dgamma, dbeta = spatial_batchnorm_backward(dx, CRP_cache[j])
            grads['cgamma' +str(i)] = dgamma
            grads['cbeta' +str(i)] = dbeta
        else:
            dx, dw, db = conv_relu_pool_backward(dx, CRP_cache[j])
        
        dw += self.reg*self.params['cW'+str(j)]
        grads['cW'+str(j)] = dw
        grads['cb'+str(j)] = db

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
