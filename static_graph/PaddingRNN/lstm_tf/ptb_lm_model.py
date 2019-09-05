import tensorflow as tf

def ptb_lm_model(hidden_size,
                 vocab_size,
                 batch_size,
                 num_layers,
                 num_steps,
                 init_scale,
                 keep_prob,
                 max_grad_norm,
                 rnn_type='static'):
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
        if rnn_type == "static":
            print( "rnn type is ", rnn_type)
            with tf.variable_scope("RNN"):
                for time_step in range(num_steps):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
    
                    (cell_output, run_state) = mul_cell(inputs[:, time_step, :], run_state)
                    outputs.append(cell_output)
            output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
        
        elif rnn_type == "padding":
            print( "rnn type is ", rnn_type)
            with tf.variable_scope("RNN"):
                output, run_state = tf.nn.dynamic_rnn( mul_cell, inputs, initial_state=run_state, time_major = False)
        
                output = tf.reshape( output, [-1, hidden_size])
        else:
            print( "not support rnn type", rnn_type)
            return Exception( "not supprt rnn type")

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


    feeding_list = [x_place, y_place, init_h_place, init_c_place]
    return cost, final_h, final_c, train_op, new_lr, lr_update, feeding_list
