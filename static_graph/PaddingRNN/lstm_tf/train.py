import time
import numpy as np
import logging

from args import *
from ptb_lm_model import *
import reader

# from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

def main():
    args = parse_args()
    model_type = args.model_type

    logger = logging.getLogger("ptb")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

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
        print("type not support", model_type)
        exit()

    if args.max_epoch > 0:
        max_epoch = args.max_epoch

    if args.profile:
        print("\nProfiler is enabled, only 1 epoch will be ran (set max_epoch = 1).\n")
        max_epoch = 1

    # Create symbolic vars
    cost, final_h, final_c, train_op, new_lr, lr_update, feeding_list = ptb_lm_model(
        hidden_size, vocab_size, batch_size, num_layers, num_steps, init_scale, keep_prob, max_grad_norm,
        rnn_type = args.rnn_type)
    
    # Initialize session
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamic allocation of VRAM
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
    raw_data = reader.ptb_raw_data(data_path)
    print( "finished load data")
    
    train_data, valid_data, test_data, _ = raw_data
    
    def eval(sess, data):
        if args.inference_only:
            sess.run(init)

        batch_times = []
        start_time = time.time()

        eval_loss = 0.0
        eval_iters = 0
        eval_data_iter = reader.get_data_iter(data, batch_size, num_steps)
    
        init_h = np.zeros( (num_layers, batch_size, hidden_size), dtype='float32')
        init_c = np.zeros( (num_layers, batch_size, hidden_size), dtype='float32')
        for batch in eval_data_iter:
            x, y = batch
            feed_dict = {}
            feed_dict[feeding_list[0]] = x
            feed_dict[feeding_list[1]] = y
            feed_dict[feeding_list[2]] = init_h
            feed_dict[feeding_list[3]] = init_c

            batch_start_time = time.time()
            output = sess.run([cost, final_h, final_c], feed_dict)
            batch_times.append(time.time() - batch_start_time)
    
            train_cost = output[0]
            init_h = output[1]
            init_c = output[2]
    
            eval_loss += train_cost
            eval_iters += num_steps
    
        ppl = np.exp(eval_loss / eval_iters)

        eval_time_total = time.time() - start_time
        eval_time_run = np.sum(batch_times)

        if args.inference_only:
            print("Eval batch_size: %d; Time (total): %.5f s; Time (only run): %.5f s; ppl: %.5f" % (batch_size, eval_time_total, eval_time_run, ppl))

        return ppl, eval_time_total
    
    
    def train(sess):
        sess.run(init)

        if args.profile:
            profiler_step = 0
            profiler = model_analyzer.Profiler(graph=sess.graph)
            run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        
        total_time = 0.0
        epoch_times = []
        
        for epoch_id in xrange(max_epoch):
            batch_times = []
            epoch_start_time = time.time()
            train_data_iter = reader.get_data_iter(train_data, batch_size, num_steps)

            # assign lr, update the learning rate
            new_lr_1 = base_learning_rate * ( lr_decay ** max(epoch_id + 1 - epoch_start_decay, 0.0) )
            sess.run( lr_update, {new_lr: new_lr_1})
        
            total_loss = 0.0
            iters = 0
            batch_len = len(train_data) // batch_size
            epoch_size = ( batch_len - 1 ) // num_steps

            if args.profile:
                log_fre = 1
            else:
                log_fre = epoch_size // 10
        
            init_h = np.zeros( (num_layers, batch_size, hidden_size), dtype='float32')
            init_c = np.zeros( (num_layers, batch_size, hidden_size), dtype='float32')
        
            count = 0.0
            for batch_id, batch in enumerate(train_data_iter):
                x,y = batch
                feed_dict = {}
                feed_dict[feeding_list[0]] = x
                feed_dict[feeding_list[1]] = y
                feed_dict[feeding_list[2]] = init_h
                feed_dict[feeding_list[3]] = init_c
        
                batch_start_time = time.time()
                if args.profile:
                    output = sess.run([cost, final_h, final_c, train_op], feed_dict, options=run_options, run_metadata=run_metadata)
                    profiler.add_step(step=profiler_step, run_meta=run_metadata)
                    profiler_step = profiler_step + 1
                    if batch_id >= 10:
                        break
                else:
                    output = sess.run([cost, final_h, final_c, train_op], feed_dict)
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
        
                train_cost = output[0]
                init_h = output[1]
                init_c = output[2]
        
                total_loss += train_cost
                iters += num_steps
                count = count + 1
                if batch_id > 0 and  batch_id % log_fre == 0:
                    ppl = np.exp( total_loss / iters )
                    print("-- Epoch:[%d]; Batch:[%d]; Time: %.5f s; ppl: %.5f, lr: %.5f" % (epoch_id, batch_id, batch_time, ppl, new_lr_1))
        
            ppl = np.exp(total_loss / iters)
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            total_time += epoch_time
        
            print("\nTrain epoch:[%d]; epoch Time: %.5f s; ppl: %.5f; avg_time: %.5f steps/s\n"
                  % (epoch_id, epoch_time, ppl, (batch_id + 1) / sum(batch_times)))

            valid_ppl, _ = eval(sess, valid_data)
            print("Valid ppl: %.5f" % valid_ppl)
    
        test_ppl, test_time = eval(sess, test_data)
        print("Test Time (total): %.5f, ppl: %.5f" % (test_time, test_ppl))
              
        if args.profile:
            profile_op_opt_builder = option_builder.ProfileOptionBuilder()
            profile_op_opt_builder.select(['micros','occurrence'])
            profile_op_opt_builder.order_by('micros')
            profile_op_opt_builder.with_max_depth(50)
            profiler.profile_operations(profile_op_opt_builder.build())


    with tf.Session(config=config) as sess:
        if not args.inference_only:
            train(sess)
        else:
            eval(sess, test_data)


if __name__ == '__main__':
    main()
