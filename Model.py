import tensorflow as tf
import numpy      as np
import math

# from NTMCell import *
from ops   import _weight_variable, _bias_variable
from utils import *
from ops   import *

class Model(object):
    def __init__(self, architecture, input_size, output_size, batch_size, time_step, LR, activation_function=None, 
                 batch_norm=True, window=1):
        # basic setting
        self.input_size   = input_size
        self.output_size  = output_size
        self.time_step    = time_step
        self.batch_size   = batch_size
        self.batch_norm   = batch_norm
        self.LR           = LR
        self.num_layer    = len(architecture)
        self.architecture = architecture
        self.window       = window
        self.sequence_length = [time_step]*batch_size # the list storing the time_step in each batch size
        self.tau          = tf.Variable(0.0, trainable=False)
        self.lr  = tf.Variable(0.0, trainable=False)
                
        # placeholder: it allow to feed in different data in each iteration
        self.x  = tf.placeholder(tf.float32, [None, input_size*window], name='x')
        self.y1 = tf.placeholder(tf.float32, [None, output_size], name='y1')
        self.y2 = tf.placeholder(tf.float32, [None, output_size], name='y2')
        self.new_tau = tf.placeholder(tf.float32, shape=[], name="new_tau")
        self.tau_update = tf.assign(self.tau, self.new_tau)
        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self.lr_update = tf.assign(self.lr, self.new_lr)
        self.training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        self.is_batch_norm_train = tf.placeholder(tf.bool)
        
        # feed forward
        with tf.variable_scope('FushionModel'):
            self.feed_forward(activation_function)
    
        # optimization
        self.compute_cost()
        self.optimizer = tf.train.AdamOptimizer(self.LR)
#         self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        
        """ not global"""
        grad_var       = self.optimizer.compute_gradients(self.cost)
        def GradientClip(grad):
            if grad is None:
                return grad
#             return tf.clip_by_norm(grad, 5)
            return tf.clip_by_value(grad, -1, 1)
        clip_grad_var = [(GradientClip(grad), var) for grad, var in grad_var ]
        self.train_op = self.optimizer.apply_gradients(clip_grad_var)
        
    def feed_forward(self, activation_function=None):
        data    = tf.reshape(self.x, [-1, self.input_size*self.window])
        self.Neurons = {'h0':data}
        self.States  = {}
        self.init_state = {}
        for idx in range(1, self.num_layer):
            if self.architecture['l'+str(idx)]['type'] == 'fc':
                now_size  = self.architecture['l'+str(idx-1)]['neurons']
                next_size = self.architecture['l'+str(idx)]['neurons']
                with tf.variable_scope('l'+str(idx)):
                    W = _weight_variable([now_size, next_size])
                    b = _bias_variable([next_size,])
                neurons = tf.nn.bias_add( tf.matmul(self.Neurons['h'+str(idx-1)], W), b )
                if activation_function != None:
                    neurons = activation_function(neurons)
                
                neurons = tf.nn.dropout(neurons, self.keep_prob)
                
                self.Neurons.update({'h'+str(idx):neurons})
                
            elif self.architecture['l'+str(idx)]['type'] == 'lstm':
                now_size  = self.architecture['l'+str(idx-1)]['neurons']
                next_size = self.architecture['l'+str(idx)]['neurons']
                self.init_state.update({'h'+str(idx):tf.zeros([self.batch_size,next_size*2])})
                neurons = []
                _input  = tf.reshape(self.Neurons['h'+str(idx-1)], [-1, self.time_step, now_size], )
                state   = self.init_state['h'+str(idx)]
                with tf.variable_scope('l'+str(idx)) as scope:
                    for time_step in range(self.time_step):
                        if time_step>0:
                            scope.reuse_variables()
                        old_h, old_c = tf.split(value=state, num_split=2, split_dim=1)
                        gates = linear(tf.concat(1, [_input[:,time_step,:], old_h]), 4*next_size)
                        i, g, f, o = tf.split(value=gates, num_split=4, split_dim=1)
                        c = tf.multiply(tf.sigmoid(f), old_c) + tf.multiply(tf.sigmoid(i), tf.tanh(g))
                        h = tf.multiply(tf.sigmoid(o), tf.tanh(c))
                        state = tf.concat(1, [h, c])
                        neurons.append(h)
                final_state = state
                neurons = tf.transpose(neurons,[1,0,2])
                neurons = tf.reshape(neurons, [-1, next_size])
    
                neurons = tf.nn.dropout(neurons, self.keep_prob)
        
                self.Neurons.update({'h'+str(idx):neurons})
                self.States.update({'h'+str(idx):final_state})
                
            elif self.architecture['l'+str(idx)]['type'] == 'mrnn':
                now_size  = self.architecture['l'+str(idx-1)]['neurons']
                next_size = self.architecture['l'+str(idx)]['neurons']
                K = self.architecture['l'+str(idx)]['state_size']
                self.init_state.update({'h'+str(idx):tf.zeros([self.batch_size,next_size*2])})
                neurons = []
                self.ALL_z = []
                self.ALL_qz = []
                _input  = tf.reshape(self.Neurons['h'+str(idx-1)], [-1, self.time_step, now_size], )
                state   = self.init_state['h'+str(idx)]
                with tf.variable_scope('l'+str(idx)) as scope:
                    for time_step in range(self.time_step):
                        if time_step>0:
                            scope.reuse_variables()
                        old_h, old_c = tf.split(value=state, num_split=2, split_dim=1)
                        logit_z = linear(tf.concat(1, [_input[:,time_step,:], old_h]), K, name='logit_encoder')
                        q_z = tf.nn.softmax(logit_z)
                        z = tf.cond(self.training, lambda: gumbel_softmax(logit_z,self.tau,hard=False), lambda: softmax_sample(logit_z))
                        z = gumbel_softmax(logit_z,self.tau,hard=False)
                        self.ALL_z.append(z)
                        self.ALL_qz.append(q_z)
                        gates = linear(tf.concat(1, [_input[:,time_step,:], old_h]), 4*K*next_size)
                        i, g, f, o = tf.split(value=gates, num_split=4, split_dim=1)
                        old_c_K = tf.tile(old_c,[1,K])
                        new_c_K = tf.multiply(tf.sigmoid(f), old_c_K) + tf.multiply(tf.sigmoid(i), tf.tanh(g))
                        new_h_K = tf.multiply(tf.sigmoid(o), tf.tanh(new_c_K))
                        new_c_K = tf.reshape(new_c_K, [self.batch_size,K,next_size])
                        new_h_K = tf.reshape(new_h_K, [self.batch_size,K,next_size])
                        h = tf.einsum('nkd,nk->nd',new_h_K,z)
                        c = tf.einsum('nkd,nk->nd',new_c_K,z)
                        state = tf.concat(1, [h, c])
                        neurons.append(h)
                final_state = state
                neurons = tf.transpose(neurons,[1,0,2])
                neurons = tf.reshape(neurons, [-1, next_size])
                
                neurons = tf.nn.dropout(neurons, self.keep_prob)
                
                self.Neurons.update({'h'+str(idx):neurons})
                self.States.update({'h'+str(idx):final_state})
                
            elif self.architecture['l'+str(idx)]['type'] == 'output':
                now_size  = self.architecture['l'+str(idx-1)]['neurons']
                next_size = self.architecture['l'+str(idx)]['neurons']
                with tf.variable_scope('output'):
                    with tf.variable_scope('sp1'):
                        W1 = _weight_variable([now_size, next_size])
                        b1 = _bias_variable([next_size,])
                    with tf.variable_scope('sp2'):    
                        W2 = _weight_variable([now_size, next_size])
                        b2 = _bias_variable([next_size,])
                #[:, ((self.window+1)/2-1)*self.input_size:((self.window+1)/2)*self.input_size]
                neurons1 = tf.nn.bias_add(tf.matmul(self.Neurons['h'+str(idx-1)], W1), b1)
                neurons2 = tf.nn.bias_add(tf.matmul(self.Neurons['h'+str(idx-1)], W2), b2)
                
                summ     = tf.add(tf.abs(neurons1), tf.abs(neurons2)) + (1e-6)
                mask1    = tf.div(tf.abs(neurons1), summ)
                mask2    = tf.div(tf.abs(neurons2), summ)
                self.pred1 = tf.mul(
                    self.Neurons['h0'][:,:self.input_size], mask1)
                self.pred2 = tf.mul(
                    self.Neurons['h0'][:,:self.input_size], mask2)
                self.Neurons.update({'h'+str(idx)+'1':self.pred1})
                self.Neurons.update({'h'+str(idx)+'2':self.pred2})
    
    def init_state_assign(self):
        self.init_state = self.States
                
    def compute_cost(self):
        self.cost_to_show = (self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'1'], self.y1) + \
                     self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'2'], self.y2))/2
        self.cost = (self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'1'], self.y1) + \
                     self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'2'], self.y2) - \
                     0*(self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'1'], self.y2) + \
                          self.ms_error(self.Neurons['h'+str(self.num_layer-1)+'2'], self.y1)))/2
    def ms_error(self, y_pre, y_target):
        return tf.reduce_sum(tf.reduce_sum( tf.square(tf.sub(y_pre, y_target)), 1))
    
    def assign_tau(self, session, tau_value):
        session.run(self.tau_update, feed_dict={self.new_tau: tau_value})
        
    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})