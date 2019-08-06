from __future__ import print_function
import numpy as np
from PIL import Image
import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer, MomentumOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC
from paddle.fluid.dygraph.base import to_variable
from benchmark import AverageMeter, ProgressMeter, Tools
import sys
import argparse


class SimpleImgConvPool(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=1,
                 act=None,
                 use_cudnn=True,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__(name_scope)
        self._conv2d = Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            act=act,
            use_cudnn=use_cudnn)
        
        self._pool2d = Pool2D(
            self.full_name(),
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)
    def forward(self, inputs):
        #print( "inp", inputs)
        x = self._conv2d(inputs)
        #print( "conv2d", x)
        x = fluid.layers.relu( x )
        #print( "relu", x)
        x = self._pool2d(x)
        #print( "pool2d", x)
        return x
class MNIST(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(MNIST, self).__init__(name_scope)
        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            self.full_name(), 20, 5, 2, 2, act="relu")
        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            self.full_name(), 50, 5, 2, 2, act="relu")
        pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (pool_2_shape**2 * SIZE))**0.5
        self._fc1 = FC(self.full_name(), 500 )
        self._fc2 = FC(self.full_name(), 10)
    def forward(self, inputs, label=None):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        #print( "second x", x)
        x = fluid.layers.reshape( x, shape=[-1, 4 * 4 * 50])
        x = self._fc1(x)
        
        x = fluid.layers.relu( x )
        x = self._fc2( x )
        x = fluid.layers.softmax( x )
        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            return x, acc
        else:
            return x
def normalize( inp, mean, std ):
    return ( inp - mean ) / std
def test_train(reader, model, batch_size):
    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(reader()):
        dy_x_data = np.array(
            [x[0].reshape(1, 28, 28)
             for x in data]).astype('float32')
        dy_x_data = normalize( dy_x_data, 0.1307, 0.3081)
        y_data = np.array(
            [x[1] for x in data]).astype('int64').reshape(batch_size, 1)
        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label.stop_gradient = True
        prediction, acc = model(img, label)
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = fluid.layers.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))
        # get test acc and loss
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()
    return avg_loss_val_mean, acc_val_mean
def test_p( reader, model, batch_size):
    total_same = 0
    total_sample = 0
    for batch_id, data in enumerate(reader()):
        dy_x_data = np.array(
            [x[0].reshape(1, 28, 28)
             for x in data]).astype('float32')
        y_data = np.array(
            [x[1] for x in data]).astype('int64').reshape(batch_size, 1)
    
        dy_x_data = normalize( dy_x_data, 0.1307, 0.3081)
        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label.stop_gradient = True
        prediction, acc = model(img, label) 
        # get final_res
        logits_p = prediction.numpy()
        
        #print( "logits p", logits_p )
        result = np.argsort( logits_p, axis=-1)[:, -1]
        
        flatten_y = y_data.reshape( -1 )
        #print( "result", result )
        #print( "y data", y_data )
        
        check_1 = np.equal( result, flatten_y ).astype( 'float32')
        #print( "check_1", check_1)
        same_num = np.sum( check_1 )
        #print( same_num, check_1.shape[0] )
        total_same += same_num
        total_sample += check_1.shape[0]
    
    print( "tatal_sampe", total_same, total_sample)
    print( "precision", total_same * 1.0 / total_sample )
def train_mnist():
    epoch_num = 10
    if args.benchmark:
        epoch_num = 1
    BATCH_SIZE = 256
    with fluid.dygraph.guard():
        mnist = MNIST("mnist")
        #adam = AdamOptimizer(learning_rate=0.001)
        adam = MomentumOptimizer( learning_rate=0.01, momentum=0.5)
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)
        eval_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=10, drop_last=True)
        for epoch in range(epoch_num):
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            progress = ProgressMeter(len(list(train_reader())) - 1, batch_time, data_time,
                                     losses, prefix="epoch: [{}]".format(epoch))
            end = Tools.time()
            for batch_id, data in enumerate(train_reader()):
                data_time.update(Tools.time() - end)
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                dy_x_data = normalize( dy_x_data, 0.1307, 0.3081)
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)
                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True
                cost, acc = mnist(img, label)
                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                batch_time.update(Tools.time() - end)
                dy_out = avg_loss.numpy()[0]
                losses.update(dy_out, BATCH_SIZE)
                if batch_id % 10 == 0:
                    progress.print(batch_id)
                end = Tools.time()
                #if batch_id % 100 == 0:
                #    print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
            mnist.eval()
            test_cost, test_acc = test_train(test_reader, mnist, BATCH_SIZE)
            test_p( eval_reader, mnist, 10)
            mnist.train()
            print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(epoch, test_cost, test_acc))
        fluid.dygraph.save_persistables(mnist.state_dict(), "save_dir")
        print("checkpoint saved")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", action="store_true", help="turn on benchmark")
    args = parser.parse_args()
    train_mnist()
