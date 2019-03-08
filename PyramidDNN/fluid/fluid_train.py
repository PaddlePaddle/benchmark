from __future__ import print_function

import glob
import numpy as np
import os
import six
import time
import multiprocessing
import threading as th

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler

from fluid_pyramid_dnn_net import Pyramidnn
import reader
from args import *
from debug_tool import *

def train_thread(idx, use_cuda, num_pass, train_py_reader, train_exe, model, logger, args):
    for epoch in six.moves.xrange(num_pass):
        #  train_py_reader.start()
        count = 0
        total_time = 0.0
        total_loss = .0

        while True:
            try:
                t1 = time.time()

                cost = train_exe.run(feed=None, fetch_list=[])

                total_time += time.time() - t1
                #  total_loss += np.array(cost[0]).mean()
                count += 1

                if count % 100 == 0:
                    print_para(model.train_main_program, train_exe, logger, model.optimizer, args)
                    print('queue length: {}'.format(train_py_reader.queue.size()))
                    print('data part: {}, batch id: {}, time: {}, loss: {}'.format(idx, count, total_time / count, total_loss / count))
            except fluid.core.EOFException:
                #  train_py_reader.reset()
                print("catch EOF")
                break

        #  train_py_reader.reset()

def raw_reader(all_files, batch_size):
    def _impl_():
	x = 0
	raw_list = []
	for file_name in all_files:
            print('reading file ' + file_name)
	    file_reader = reader.train_file_read(file_name)
	    for data in file_reader():
		if x == batch_size:
		    yield raw_list
                    raw_list = []
		    x = 0
		else:
	            raw_list.append(data)
		    x += 1
    return _impl_

def get_train_multiprocess_reader(reader, datasets, process_num=10):
    groups = []
    process_num = min(process_num, len(datasets))
    for i in range(process_num):
        groups.append(datasets[i::process_num])

    readers = []
    for group in groups:
        readers.append(reader(group))

    return paddle.reader.multiprocess_reader(readers, use_pipe=True)

def train(num_pass=1, use_cuda=False, mem_opt=False):
    args = parse_args()
    import logging 
    logger = logging.getLogger("pyramid")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.info('Running with args : {}'.format(args))

    dict_size = 750000
    hash_size = 750000
    print_iter = 1000
    eval_iter = 6000
    batch_size = args.batch_size
    thread_num = int(os.environ.get('THREAD_NUM', 1))
    print('training data parallel num: {}'.format(thread_num))
    debug = False
    file_pattern='../traindata_pyramid_20180204_ids_wise_1month_test/part-0000*'
    #file_pattern='td_6_0001'

    np.random.seed = 1

    # construct network
    model = Pyramidnn(hash_size=hash_size, dict_size=dict_size)
    pos_sim_name, train_main_program, train_startup_program, train_py_reader = model.build() 

    # memory optimize
    if mem_opt:
        fluid.memory_optimize(train_main_program)

    if args.para_print:
        debug_init(train_main_program, model.vars, model.vars_name)
    # initialize parameters
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(train_startup_program)

    print("reading data")

    all_files = glob.glob(file_pattern)
    
    my_reader = lambda x:raw_reader(x, batch_size=batch_size)
    multi_reader = get_train_multiprocess_reader(my_reader, all_files)
    train_py_reader.decorate_paddle_reader(multi_reader)
    train_py_reader.start()

    print("reading end")

    start = time.time()
    th_list = []

    for i in range(thread_num):
        local_scope = fluid.global_scope().new_scope()
        # PE config
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_cuda = use_cuda # CPU
        exec_strategy.use_experimental_executor = True # experimental executor
        exec_strategy.allow_op_delay = True
        exec_strategy.num_threads = 1 # thread num
        exec_strategy.num_iteration_per_drop_scope = 100000
        build_strategy = fluid.BuildStrategy()
        build_strategy.remove_unnecessary_lock = True
        pass_builder = build_strategy._finalize_strategy_and_create_passes()
        pass_builder.insert_pass(0, "lock_free_optimize_pass")
        train_exe = fluid.ParallelExecutor(
                use_cuda=use_cuda,
                loss_name=None,
                main_program=train_main_program,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy,
                scope=local_scope)

        t = th.Thread(target=train_thread, args=(i, use_cuda, num_pass, train_py_reader, train_exe, model, logger, args))
        t.start()
        th_list.append(t)

    for t in th_list:
        t.join()

    print('total time: {}'.format(time.time() - start))

    test_program, pos_sim_var = model.build_test(hash_size=hash_size, dict_size=dict_size, parallelism=1)
    pos_sim_var_name = pos_sim_var.name
    test_feed_var_names = ['query_basic', 'query_phrase', 'pos_title_basic', 'pos_title_phrase']
    test_feed_list = [test_program.global_block().var(var_name) for var_name in test_feed_var_names]
    test_feeder = fluid.DataFeeder(test_feed_list, fluid.CPUPlace())

    test_exe = fluid.ParallelExecutor(
            use_cuda=use_cuda,
            main_program=test_program,
            scope=fluid.global_scope(),
            share_vars_from=train_exe)

    print('start evaluate')
    t2 = time.time()
    with open('./eval_result/' + paddle.version.commit + '_' + str(time.time()), 'w') as f:
        test_batch_reader = paddle.batch(
                reader.test(test_file='test_data'),
                batch_size=1280)

        for test_data in test_batch_reader():
            qids = []
            labels = []
            data_list = []
            for one_data in test_data:
                qids.append(one_data[0])
                labels.append(int(one_data[-1][0]))
                data_list.append(one_data[1:-1])
            predicts = test_exe.run(feed=test_feeder.feed(data_list), fetch_list=[pos_sim_var_name])
            scores = np.array(predicts[0])

            for qid, label, score in six.moves.zip(qids, labels, scores):
                f.write(str(qid) + '\t' + str(score[0]) + '\t' + str(label) + '\n')

    print('end eval', time.time() - t2)

    print('start save model')

    fluid.io.save_inference_model('model', test_feed_var_names, [pos_sim_var], train_exe)

if __name__ == '__main__':
    train()

