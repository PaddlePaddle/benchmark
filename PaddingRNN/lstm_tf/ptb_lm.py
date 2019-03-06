import os
import io
import sys
import time

import tensorflow as tf

import reader

import numpy as np
# from tensorflow.python.client import timeline
# from tensorflow.python.profiler import model_analyzer
# from tensorflow.python.profiler import option_builder
# Get data

model_type = "small"
inference_only = False
if len(sys.argv) >= 2:
    model_type = sys.argv[1]
    inference_only = sys.argv[2]

if model_type == "small":
    batch_size = 20
    num_steps = 20
    hidden_size = 200
    num_layers = 2
    vocab_size = 10000
    keep_prob = 1.0
    init_scale = 0.1
    max_grad_norm = 5
    max_epoch = 13
    base_learning_rate = 1.0
    lr_decay = 0.5
    epoch_start_decay = 4 
elif model_type == "medium":
    batch_size = 20
    num_steps = 35
    hidden_size = 650
    num_layers = 2
    vocab_size = 10000
    keep_prob = 0.5
    init_scale = 0.05
    max_grad_norm = 5
    max_epoch = 39
    base_learning_rate = 1.0
    lr_decay = 0.8
    epoch_start_decay = 6
elif model_type == "large":
    batch_size = 20
    num_steps = 35
    hidden_size = 1500
    num_layers = 2
    vocab_size = 10000
    keep_prob = 0.35
    init_scale = 0.04
    max_grad_norm = 10
    max_epoch = 55
    base_learning_rate = 1.0
    lr_decay = 1/1.15
    epoch_start_decay = 14
else:
    print( "type not support", model_type)

    exit()
# Create symbolic vars


x_place = tf.placeholder(tf.int32, [batch_size, num_steps])
y_place = tf.placeholder(tf.int32, [batch_size, num_steps])
init_h_place = tf.placeholder(tf.float32, [num_layers, batch_size, hidden_size])
init_c_place = tf.placeholder(tf.float32, [num_layers, batch_size, hidden_size])



initializer = tf.random_uniform_initializer(-init_scale, init_scale)

with tf.variable_scope("Model", reuse=None, initializer=initializer):
    embedding = tf.get_variable(
            "embedding", [vocab_size, hidden_size], dtype=tf.float32)

    inputs = tf.nn.embedding_lookup(embedding, x_place)

    if keep_prob < 1:
        inputs = tf.nn.dropout(inputs, keep_prob)

    cell_list = []

    # build state
    states = []
    for i in range(num_layers):
        s = tf.contrib.rnn.LSTMStateTuple(h=init_h_place[i], c=init_c_place[i])
        states.append(s)

    run_state = tuple(states)

    for i in range(num_layers):
        cell = tf.contrib.rnn.BasicLSTMCell(
            hidden_size, forget_bias=0.0, state_is_tuple=True )
        if keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=keep_prob)
        cell_list.append( cell )
    mul_cell = tf.contrib.rnn.MultiRNNCell( cell_list, state_is_tuple=True )

    outputs = []
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()

            (cell_output, run_state) = mul_cell(inputs[:, time_step, :], run_state)
            outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])

    final_h_arr = []
    final_c_arr = []

    for i in range(num_layers):
        final_h_arr.append( run_state[i].h)
        final_c_arr.append( run_state[i].c)

    final_h = tf.concat( final_h_arr, 0 )
    final_h = tf.reshape( final_h, [num_layers, -1, hidden_size])
    final_c = tf.concat( final_c_arr, 0)
    final_c = tf.reshape( final_c, [num_layers, -1, hidden_size])

    softmax_w = tf.get_variable(
            "softmax_w", [hidden_size, vocab_size], dtype='float32')
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype='float32')

    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])

    loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            y_place,
            tf.ones([batch_size, num_steps], dtype='float32'),
            average_across_timesteps=False,
            average_across_batch=True)
    cost = tf.reduce_sum(loss)

    lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                        max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

    new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
    lr_update = tf.assign(lr, new_lr)

# Initialize session
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamic allocation of VRAM
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = False  # dynamic allocation of VRAM

# Print parameter count
params = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    params += variable_parameters
print('# network parameters: ' + str(params))

data_path = "data/simple-examples/data"
raw_data = reader.ptb_raw_data( data_path )
print( "finished load data")

train_data, valid_data, test_data, _ = raw_data

data_len = len(train_data)

with tf.Session(config=config) as sess:
    sess.run(init)
    # Check for correct sizes
    # my_profiler = model_analyzer.Profiler(graph=sess.graph)
    # run_metadata = tf.RunMetadata()
    start = time.time()
    for epoch_id in xrange( max_epoch ):
        train_data_iter = reader.get_data_iter( train_data, batch_size, num_steps)
        # assign lr
        new_lr_1 = base_learning_rate * ( lr_decay ** max(epoch_id + 1 - epoch_start_decay, 0.0) )
        sess.run( lr_update, {new_lr: new_lr_1})

        total_loss = 0.0
        iters = 0
        batch_len = len(train_data) // batch_size
        epoch_size = ( batch_len - 1 ) // num_steps
        log_fre = epoch_size // 10

        def eval( data ):
            st=time.time()
            eval_loss = 0.0
            eval_iters = 0
            eval_data_iter = reader.get_data_iter( data, batch_size, num_steps)

            init_h = np.zeros( (num_layers, batch_size, hidden_size), dtype='float32')
            init_c = np.zeros( (num_layers, batch_size, hidden_size), dtype='float32')
            for batch in eval_data_iter:
                x, y = batch
                feed_dict = {}
                feed_dict[x_place] = x
                feed_dict[y_place] = y
                feed_dict[init_h_place] = init_h
                feed_dict[init_c_place] = init_c
                output = sess.run( [cost, final_h, final_c], feed_dict )

                train_cost = output[0]
                init_h = output[1]
                init_c = output[2]

                eval_loss += train_cost
                eval_iters += num_steps

            ppl = np.exp( eval_loss / eval_iters )
            ed = time.time()
            print("test time is {}".format(ed-st))
            return ppl

        init_h = np.zeros( (num_layers, batch_size, hidden_size), dtype='float32')
        init_c = np.zeros( (num_layers, batch_size, hidden_size), dtype='float32')


        if not inference_only:
            count = 0.0
            for batch_id, batch in enumerate(train_data_iter):

                x,y = batch
                feed_dict = {}
                feed_dict[x_place] = x
                feed_dict[y_place] = y
                feed_dict[init_h_place] = init_h
                feed_dict[init_c_place] = init_c
                t1 = time.time()
                output = sess.run( [cost, final_h, final_c, train_op], feed_dict )


                train_cost = output[0]
                init_h = output[1]
                init_c = output[2]

                total_loss += train_cost
                iters += num_steps
                count  = count + 1
                if batch_id >0 and  batch_id % log_fre == 0:
                    ppl = np.exp( total_loss / iters )
                    print( "ppl ", batch_id, ppl, new_lr_1 )




            ppl  = np.exp( total_loss / iters )

            print( "epoch", epoch_id, "loss", ppl)

            val_ppl = eval( valid_data )

            print( "eval ppl", val_ppl)

    start = time.time()

    test_ppl = eval( test_data )
    #
    end = time.time()
    print("total time is {}".format(end - start))
    print( "test ppl", test_ppl)
            

            
