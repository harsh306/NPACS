import numpy as np
import tensorflow as tf
import ops2
import sys
import matplotlib.pyplot as plt
import collections
import math
import sys
import time
import os


sys.path.append(os.getcwd())


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class NPCS_AE:
    def __init__(self, config):
        self.X, self.d_dim, self.code_dim = ops2.get_data(config.data, config.fill_points, 1.0)
        self.config = config
        self.limit = 1.0
        self.ev = None 
        
    def positive(self,l):
        l = tf.assign(l,self.limit)
        return l

    def neg(self,l,delta_l):
        l = tf.assign(l, l+delta_l )
        l = tf.cond(l >= 1.0, true_fn= lambda: self.positive(l),false_fn= lambda: l ) 
        return l

    def update_l(self,l,delta_l):
        l = tf.cond(l >= 1.0, true_fn= lambda: self.positive(l), false_fn= lambda: self.neg(l,delta_l) )    
        return l


    def autoencoder8_svd(self,x,l):
        with tf.variable_scope('network'):
            encoder =  ops2.activation(self.config.use_act, ops2.etlinear2(x,200,self.ev,scope ='encoder1'),l )
            encoder = ops2.activation(self.config.use_act, ops2.etlinear2(encoder,100,self.ev,scope ='encoder2'),l )
            encoder = ops2.activation(self.config.use_act, ops2.etlinear2(encoder,50,self.ev,scope ='encoder3'),l)
            code =  ops2.etlinear2(encoder,2,self.ev,scope ='code')
            decoder = ops2.activation(self.config.use_act, ops2.etlinear2(code,50,self.ev,scope ='decoder3'),l)
            decoder = ops2.activation(self.config.use_act, ops2.etlinear2(decoder,100,self.ev,scope ='decoder2'),l)
            decoder = ops2.activation(self.config.use_act, ops2.etlinear2(decoder,200,self.ev,scope ='decoder1'),l)
            out =  ops2.etlinear2(decoder,784,self.ev,scope ='output')
            loss = tf.reduce_mean(tf.square(x - out))
            return loss, out, code

    def autoencoder16_svd(self,x,l):
        with tf.variable_scope('network'):
            encoder = ops2.activation(self.config.use_act, ops2.stlinear2(x,500,self.ev,scope ='encoder1'),l )
            encoder = ops2.activation(self.config.use_act, ops2.stlinear2(encoder,200,self.ev,scope ='encoder2'),l )
            encoder = ops2.activation(self.config.use_act, ops2.stlinear2(encoder,100,self.ev,scope ='encoder3'),l)
            encoder = ops2.activation(self.config.use_act, ops2.stlinear2(encoder,50,self.ev,scope ='encoder4'),l)
            encoder = ops2.activation(self.config.use_act, ops2.stlinear2(encoder,50,self.ev,scope ='encoder5'),l)
            encoder = ops2.activation(self.config.use_act, ops2.stlinear2(encoder,5,self.ev,scope ='encoder6'),l)
            encoder = ops2.activation(self.config.use_act, ops2.stlinear2(encoder,5,self.ev,scope ='encoder7'),l)
            code = ops2.stlinear2(encoder,2,self.ev,scope ='code')
            decoder = ops2.activation(self.config.use_act, ops2.stlinear2(code,5,self.ev,scope ='decoder7'),l)
            decoder = ops2.activation(self.config.use_act, ops2.stlinear2(decoder,5,self.ev,scope ='decoder6'),l)
            decoder = ops2.activation(self.config.use_act, ops2.stlinear2(decoder,50,self.ev,scope ='decoder5'),l)
            decoder = ops2.activation(self.config.use_act, ops2.stlinear2(decoder,50,self.ev,scope ='decoder4'),l)
            decoder = ops2.activation(self.config.use_act, ops2.stlinear2(decoder,100,self.ev,scope ='decoder3'),l)
            decoder = ops2.activation(self.config.use_act, ops2.stlinear2(decoder,200,self.ev,scope ='decoder2'),l)
            decoder = ops2.activation(self.config.use_act, ops2.stlinear2(decoder,500,self.ev,scope ='decoder1'),l)
            out = ops2.stlinear2(decoder,784,self.ev,scope ='output')
            loss = tf.reduce_mean(tf.square(x - out))
            return loss, out, code

            
    def master_graph(self):
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, shape=[None,self.d_dim], name='x')
        l = tf.Variable(self.config.l_init, dtype=tf.float32,trainable=False,name='lamda')
        
        delta_l = tf.placeholder(dtype=tf.float32,shape=[],name='delta_l' )
        l_prev = tf.placeholder( dtype=tf.float32,shape=[],name='lamda_prev')
        omega = tf.placeholder(dtype=tf.float32,shape=[],name='omega' )
        lnorm = tf.placeholder(dtype=tf.float32,shape=[],name='lnorm' )
        
        if self.config.svd == True:
            self.X = ops2.center_data(self.X)
 
            if self.config.depth == 8:
                self.ev = ops2.get_v_n(self.X)

                with tf.variable_scope('current'):
                    loss_c, output_c, code = self.autoencoder8_svd(x,l)
                with tf.variable_scope('prev'):
                    loss_p, output_p, _ = self.autoencoder8_svd(x,l)
                with tf.variable_scope('prev2'):
                    loss_p2, output_p2, _ = self.autoencoder8_svd(x,l)
            else:
                self.ev = ops2.get_v_n16(self.X)

                with tf.variable_scope('current'):
                    loss_c, output_c, code = self.autoencoder16_svd(x,l)
                with tf.variable_scope('prev'):
                    loss_p, output_p, _ = self.autoencoder16_svd(x,l)
                with tf.variable_scope('prev2'):
                    loss_p2, output_p2, _ = self.autoencoder16_svd(x,l)

        
        optimizer =  tf.train.RMSPropOptimizer(self.config.lr)
        grads_and_vars = optimizer.compute_gradients(loss_c)
        opt = optimizer.apply_gradients(grads_and_vars)
        grads, _ = list(zip(*grads_and_vars))
        norms = tf.global_norm(grads)

        #lambda update NPC
        l_new = self.update_l(l,delta_l)
        
        #secant update for lambda NPCS
        A = tf.trainable_variables(scope='current/network')
        B = tf.trainable_variables(scope='prev/network')
        C = tf.trainable_variables(scope='prev2/network')
        
        copy_op = ops2.copy_g(A,B)
        copy_op1 = ops2.copy_g(A,C)
        copy_op2 = ops2.copy_g(C,B)
        
        diff_op = ops2.diff_l(A,B,self.config)
        secant_op = ops2.secant_l(A,B,self.config)
        
        return AttrDict(locals())      

        

   

        
 
