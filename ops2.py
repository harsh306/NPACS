import itertools
import pickle

import matplotlib
import numpy as np
import tensorflow as tf
from sklearn import datasets
# from pythonsmote.SMOTE import SMOTE
from tensorflow.examples.tutorials.mnist import input_data

matplotlib.use('Agg')
import sys
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd


def save_plots(code, losses, val_losses, lamdas, norms, config):
    def _get_labels():
        if config.data == 'cifar':
            _, labels = CIFAR_data()
            labels = labels[:40000]
        else:
            data = input_data.read_data_sets('data/fashion')
            labels = data.train.labels
        return labels

    labels = _get_labels()
    final = np.column_stack((code[:, 0:2], np.asarray(labels)))
    final_df = pd.DataFrame(final, columns=['pc1', 'pc2', 'targets'])
    final_df.head()
    fig = plt.figure(figsize = (25,25))
    ax = fig.add_subplot(2,2,1) 
    ax.set_xlabel(' Component 1', fontsize = 15)
    ax.set_ylabel(' Component 2', fontsize = 15)
    ax.set_title('Code components ', fontsize = 15)
    targets = set(labels)
    colors = cm.rainbow(np.linspace(0, 1, 10))
    for target, color in zip(targets,colors):
        indicesToKeep = final_df['targets'] == target
        ax.scatter(final_df.loc[indicesToKeep, 'pc1']
                   , final_df.loc[indicesToKeep, 'pc2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    ax2 = fig.add_subplot(2,2,2) 
    ax2.plot(losses)
    ax2.set_title('Reconstruction Loss', fontsize=15)
    ax2.set_xlabel('Global steps', fontsize=15)
    ax2.set_ylabel(' Loss', fontsize=15)
    ax3 = fig.add_subplot(2,2,3) 
    ax3.plot(lamdas)
    ax3.set_title('lambda steps', fontsize=15)
    ax3.set_xlabel('Global steps', fontsize=15)
    ax3.set_ylabel('lambda', fontsize=15)
    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(val_losses)
    ax4.set_title('val_loss', fontsize=15)
    ax4.set_xlabel('Global steps', fontsize=15)
    ax4.set_ylabel('val_loss', fontsize=15)
    fig.savefig('./results/' + str(config.omega_exp) + '/' + str(config.use_act) + '.png')
    plt.close()
    return
    

def C_SMOTE(T,fill_points,alpha,k=25):
    N = 1
    #fill_points = 50
    smote = SMOTE(T,N,k,fill_points,alpha)


    synth = smote.over_sampling()
    #print('# Synth Samps: ', synth.shape)
    return synth

def center_data(X):
    mean_x = np.mean(X, axis=0, keepdims=True)
    reduced_mean = np.subtract(X,mean_x)
    reduced_mean = reduced_mean.astype(np.float32)
    return reduced_mean


def parity_batch(input_length, batch_size):
    xs = [np.random.randint(0, 2, input_length) for _ in range(batch_size)]
    xs.append(np.ones(shape=input_length, dtype=int))
    ys = [[0] if np.sum(x) % 2 == 0 else [1] for x in xs]
    return xs, ys

def get_batch_with_labels(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def get_batch(num, data):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    #print(data.shape)
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]

    return np.asarray(data_shuffle)


def get_data(data, fill_points, a_, config):
    if data == 'sine':
        X_o = SINE_data()
        X_o = center_data(X_o)
        X = C_SMOTE(X_o,fill_points,a_)
        d_dim = 2
        code_dim = 1
        return X,X_o, d_dim, code_dim
    #X2 = C_SMOTE(X_o,fill_points,0.5)
    elif data == 'mnist':
        X = np.asarray(MNIST_data())
        #X  = X / 255.0
        d_dim = 784
        code_dim = 2
        return X, d_dim, code_dim
    elif data == 'fashion':
        X, X_val = np.asarray(FASHION_data())
        #X  = X / 255.0
        d_dim = 784
        code_dim = 2
        #X = center_data(X)
        return X, X_val, d_dim, code_dim
    elif data == 'cifar':
        X, _ = CIFAR_data()
        X = X / 255.0
        d_dim = 3072
        code_dim = 2
        X1 = X[:40000]
        X_val = X[-10000:]
        print(X1.shape, X_val.shape)
        del X
        return X1, X_val, d_dim, code_dim
    elif data == 'parity':
        d_dim = config.parity_length
        X, y = parity_batch(config.parity_length, 200000)
        X_val, y_val = parity_batch(config.parity_length, 50000)
        return X, y, X_val, y_val, d_dim
    elif data == 'swiss':
        X_o = SWISS_data()
        #X_o = 1 / (1 + np.exp(-1* X_o))
        X = C_SMOTE(X_o,fill_points,a_,k=50)
        d_dim = 3
        code_dim = 2
        return X, X_o, d_dim, code_dim
    elif data == 'grid':
        X_o = GRID_data()
        X = C_SMOTE(X_o,fill_points,a_,k=50)
        #X = 1 / (1 + np.exp(-1* X))
        d_dim = 2
        code_dim = 2
        return X, d_dim, code_dim
    else :
        sys.exit()
    
def SWISS_data():
    X = datasets.make_swiss_roll(n_samples=1000, noise=0.0, random_state=0)[0]
    return X

def MNIST_data():
    mnist = input_data.read_data_sets("./MNIST_data")
    return mnist.train.images
def FASHION_data():
    data = input_data.read_data_sets('data/fashion')
    return data.train.images, data.test.images


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    features = features.reshape((len(batch['data']), 3 * 32 * 32))
    print('f', features.shape)
    labels = batch['labels']

    return features, labels


def CIFAR_data():
    images_array, image_labels = load_cfar10_batch(cifar10_dataset_folder_path='../data/cifar', batch_id=1)
    for i in range(2, 6):
        images_array1, image_labels1 = load_cfar10_batch(cifar10_dataset_folder_path='../data/cifar', batch_id=i)
        images_array = np.concatenate((images_array, images_array1), axis=0)
        image_labels = np.concatenate((image_labels, image_labels1), axis=0)

    return images_array, image_labels


def GRID_data():
    grid = np.array([np.array([i, j]) for i, j in 
                    itertools.product(np.linspace(-2, 2, 5),
                                      np.linspace(-2, 2, 5))],dtype=np.float32)

    return grid
    
def SINE_data():
    
    fs = 100 # sample rate 
    f = 4 # the frequency of the signal

    x = np.arange(fs) # the points on the x axis for plotting
    # compute the value (amplitude) of the sin wave at the for each sample
    y = np.asarray([ np.sin(2*np.pi*f * (i/fs)) for i in x])

    X = np.stack((x,y), axis=0)
    #X = np.abs(X.T)
    X = X.T
    #print(X.shape)
    return X

def copy_g(a,b):
    temp = []
    for i,j in zip(a,b):
        k = tf.assign(j,i)
        temp.append(k)
    return temp

def diff_l(a,b,config):
    temp = []
    for i,j in zip(a,b):
        k = tf.assign(j, (i -j))
        temp.append(k)
    return temp


def secant_l(a,b,config):
    temp = []
    for i,t in zip(a,b):
        k = tf.assign(i, i + ((config.delta_l/config.delta_l_prev)*t) )
        temp.append(k)
    return temp


def secant_lambda(a,b,config):
    c  = []
    c1 = []
    norm_list = []
    for i,j in zip(a,b):
        k = tf.assign(j, (i -j))
        c.append(k)

    for i,t in zip(a,c):
        k = tf.assign(i, i + tf.multiply((config.delta_l/config.delta_l_prev), t) )
        c1.append(k)
    del c, norm_list
    return c1


def secant_g(a,b,l,lnorm,omega):
    c  = []
    c1 = []
     
    norm_list = []
    for i,j in zip(a,b):
        norm_list.append(tf.reduce_sum(tf.square(i-j)))
        k = tf.assign(j, (i -j))
        c.append(k)
        
    theta_norm = tf.add_n(norm_list)
    # or b
    for i,t in zip(a,c):
        k = tf.assign(i, i + tf.multiply( (omega ), tf.div(t , tf.sqrt(theta_norm + lnorm)  ) ) )
        #k = tf.assign(i , tf.div(i, 100))
        c1.append(k)
    del c, norm_list
    return c1, theta_norm


def secant_g2(a,b,l,lnorm,omega):
    c  = []
    c1 = []
     
    norm_list = []
    for i,j in zip(a,b):
        norm_list.append(tf.reduce_sum(tf.square(i-j)))
        k = tf.assign(j,   (i -j)  )
        c.append(k)
        
    theta_norm = tf.add_n(norm_list) #np.sum(norm_list)
    # or b
    for i,t in zip(a,c):
        k = tf.assign(i, i + tf.multiply( (omega ), tf.div(t , tf.sqrt(theta_norm + lnorm)  ) ) )
        #k = tf.assign(i , tf.div(i, 100))
        c1.append(k)
    del c, norm_list
    return c1, theta_norm



def etlinear(input_, output_size,ev,code_dim=2,n=5, scope = None, stddev=0.5, bias_start=0.0001):
    
    shape = input_.get_shape().as_list()
    
    with tf.variable_scope(scope or "Linear"):
        if scope == 'encoder1':
            w = ev.T[:,:200]
        elif scope == 'encoder2':
            w = ev.T[:200,:100]
        elif scope == 'encoder3':
            w = ev.T[:100,:50]
        elif scope == 'code':
            w = ev.T[:50,:2]
        elif scope == 'decoder3':
            w = (ev.T[:50,:2]).T
        elif scope == 'decoder2':
            w = (ev.T[:100,:50]).T
        elif scope == 'decoder1':
            w = (ev.T[:200,:100]).T
        elif scope == 'output':
            w = (ev.T[:,:200]).T
        w = np.asarray(w).reshape(shape[1],output_size)
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,   initializer=tf.constant_initializer(w))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        
        return tf.matmul(input_, matrix) + bias
    


    
def get_v_n(X):
    _,_,ev200 = np.linalg.svd(X, full_matrices=False)
    l1 = np.dot(X, ev200.T[:,:200])
    
    
    _,_,ev100 = np.linalg.svd(l1, full_matrices=False)
    l2 = np.dot(l1, ev100.T[:,:100])
    
    _,_,ev50 = np.linalg.svd(l2, full_matrices=False)
    l3 = np.dot(l2, ev50.T[:,:50])
    
    
    _,_,ev2 = np.linalg.svd(l3, full_matrices=False)
    
    ev = {'ev200':ev200,'ev100':ev100,'ev50':ev50,'ev2':ev2}
    
    return ev
    

def get_v_n16(X):
    _,_,ev500 = np.linalg.svd(X, full_matrices=False)
    l1 = np.dot(X, ev500.T[:,:500])

    _,_,ev200 = np.linalg.svd(l1, full_matrices=False)
    l2 = np.dot(l1, ev200.T[:,:200])

    _,_,ev100 = np.linalg.svd(l2, full_matrices=False)
    l3 = np.dot(l2, ev100.T[:,:100])

    _,_,ev50 = np.linalg.svd(l3, full_matrices=False)
    l4 = np.dot(l3, ev50.T[:,:50])

    _,_,ev50_ = np.linalg.svd(l4, full_matrices=False)
    l5 = np.dot(l4, ev50_.T[:,:50])

    _,_,ev5 = np.linalg.svd(l5, full_matrices=False)
    l6 = np.dot(l5, ev5.T[:,:5])

    _,_,ev5_ = np.linalg.svd(l6, full_matrices=False)
    l7 = np.dot(l6, ev5_.T[:,:5])

    _,_,ev2 = np.linalg.svd(l7, full_matrices=False)

    ev = {'ev500':ev500, 'ev200':ev200, 'ev100':ev100, 'ev50':ev50, 'ev50_':ev50_, 'ev5':ev5, 'ev5_':ev5_, 'ev2':ev2}

    return ev
    

def etlinear2(input_, output_size,ev,code_dim=2,n=5, scope = None, stddev=0.5, bias_start=0.0001):
    
    shape = input_.get_shape().as_list()
    
    with tf.variable_scope(scope or "Linear"):
        if scope == 'encoder1':
            w = ev.get('ev200').T[:,:200]
        elif scope == 'encoder2':
            w = ev.get('ev100').T[:,:100]
        elif scope == 'encoder3':
            w = ev.get('ev50').T[:,:50]
        elif scope == 'code':
            w = ev.get('ev2').T[:,:2]
        elif scope == 'decoder3':
            w = (ev.get('ev2').T[:,:2]).T
        elif scope == 'decoder2':
            w = (ev.get('ev50').T[:,:50]).T
        elif scope == 'decoder1':
            w = (ev.get('ev100').T[:,:100]).T
        elif scope == 'output':
            w = (ev.get('ev200').T[:,:200]).T
        w = np.asarray(w).reshape(shape[1],output_size)
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,   initializer=tf.constant_initializer(w))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        
        return tf.matmul(input_, matrix) + bias


def stlinear2(input_, output_size,ev,code_dim=2,n=5, scope = None, stddev=0.5, bias_start=0.0001):

    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        if scope == 'encoder1':
            w = ev.get('ev500').T[:,:500]
        elif scope == 'encoder2':
            w = ev.get('ev200').T[:,:200]
        elif scope == 'encoder3':
            w = ev.get('ev100').T[:,:100]
        elif scope == 'encoder4':
            w = ev.get('ev50').T[:,:50]
        elif scope == 'encoder5':
            w = ev.get('ev50_').T[:,:50]
        elif scope == 'encoder6':
            w = ev.get('ev5').T[:,:5]
        elif scope == 'encoder7':
            w = ev.get('ev5_').T[:,:5]
        elif scope == 'code':
            w = ev.get('ev2').T[:,:2]
        elif scope == 'decoder7':
            w = (ev.get('ev2').T[:,:2]).T
        elif scope == 'decoder6':
            w = (ev.get('ev5_').T[:,:5]).T
        elif scope == 'decoder5':
            w = (ev.get('ev5').T[:,:5]).T
        elif scope == 'decoder4':
            w = (ev.get('ev50_').T[:,:50]).T
        elif scope == 'decoder3':
            w = (ev.get('ev50').T[:,:50]).T
        elif scope == 'decoder2':
            w = (ev.get('ev100').T[:,:100]).T
        elif scope == 'decoder1':
            w = (ev.get('ev200').T[:,:200]).T
        elif scope == 'output':
            w = (ev.get('ev500').T[:,:500]).T
        w = np.asarray(w).reshape(shape[1],output_size)
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,   initializer=tf.constant_initializer(w))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias

    
## Continuous logit
def c_sigmoid(v, l):

    #print('real l', str(l))
    c =  ((1-l)*v) + (l * tf.nn.sigmoid(v))
    return c

def c_relu(v, l):
    c = ((1-l)*v) + (l * tf.nn.relu(v))
    return c

def c_tanh(v,l):
    c = ((1-l)*v) + (l * tf.nn.tanh(v))
    return c

def activation(act_key, v, l):
    if act_key == "sigmoid":
        return tf.nn.sigmoid(v)
    elif act_key == "relu":
        return tf.nn.relu(v)
    elif act_key == "c_sigmoid":
        return c_sigmoid(v, l)
    elif act_key == "c_relu":
        return c_relu(v, l)
    elif act_key == "c_tanh":
         return c_tanh(v, l)
    elif act_key == "tanh":
        return tf.nn.tanh(v)
    elif act_key =="sin":
        return tf.sin(v)
    else :
        return v
    
    
def adaptive_lambda(config,step,norms):
    if (step > config.adaptive_threshold) and (step > config.u_freq) and (step > config.adaptive_start):
        avg_p = np.mean(norms[-(2*config.adaptive_threshold):-config.adaptive_threshold])
        avg_c = np.mean(norms[-config.adaptive_threshold:])
        if avg_p == 0:
            avg_p = 0.2
        if ( (avg_c - avg_p)  / avg_p ) < -config.norm_strict:
            config.delta_l = config.delta_l*(1.5)
            #config.u_freq = config.u_freq - config.u_freq_delta
            if config.delta_l >= config.delta_l_max:
                config.delta_l = config.delta_l_max
            if config.u_freq <=config.u_freq_min:
                config.u_freq = config.u_freq_min
        elif ( (avg_c - avg_p)  / avg_p ) >= config.norm_strict:
            config.delta_l = config.delta_l/2
            #config.u_freq = config.u_freq + config.u_freq_delta
            if config.delta_l <= config.delta_l_min:
                config.delta_l = config.delta_l_min
            if config.u_freq >= config.u_freq_max:
                config.u_freq = config.u_freq_max
        else:
            pass

        if max(norms[-config.adaptive_threshold:]) >= 0.25:  # 5
            config.delta_l = config.delta_l/2
            if config.delta_l <= config.delta_l_min:
                config.delta_l = config.delta_l_min

    return config.delta_l

def adaptive(config,step,losses):
    if (step > config.adaptive_threshold) and (step > config.u_freq):
        #m_ = m_- ( np.sign((closs - sum(losses[-l_freq:])/l_freq )/closs) )* 0.10
        avg_p = np.mean(losses[-(2*config.adaptive_threshold):-config.adaptive_threshold])
        avg_c = np.mean(losses[-config.adaptive_threshold:])
        if ( (avg_c - avg_p)  / avg_p ) < -config.loss_strict:
            config.omega = config.omega +  config.omega_delta
            config.u_freq = config.u_freq - config.u_freq_delta
            if config.omega >= config.omega_max:
                config.omega = config.omega_max
            if config.u_freq <=config.u_freq_min:
                config.u_freq = config.u_freq_min
        elif ( (avg_c - avg_p)  / avg_p ) >= config.loss_strict:
            config.omega = config.omega - config.omega_delta
            config.u_freq = config.u_freq + config.u_freq_delta
            if config.omega <= config.omega_min:
                config.omega = config.omega_min
            if config.u_freq >= config.u_freq_max:
                config.u_freq = config.u_freq_max
        else:
            pass
             
    return config.omega, config.u_freq
