import numpy as np
import os
import six
import time
import multiprocessing

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler

from pyramid_dnn_net import net
from pyramid_dnn_net import test_net
import reader

def train(num_pass=300, use_cuda=False, mem_opt=False):
    dict_size = 100000
    hash_size = 100000
    print_iter = 100
    eval_iter = 1
    batch_size = 128
    cpu_num = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    debug = False

    fluid.default_startup_program().random_seed = 1
    fluid.default_main_program().random_seed = 1
    np.random.seed = 1

    # construct network
    loss, pos_sim, train_program, test_program = net(hash_size=hash_size, dict_size=dict_size)

    #  optimizer = fluid.optimizer.Adam(learning_rate=1e-4)
    optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
    optimizer.minimize(loss)

    # memory optimize
    if mem_opt:
        fluid.memory_optimize(fluid.default_main_program())

    for var in train_program.blocks[0].vars:
        #  if "GRAD" not in var and not train_program.blocks[0].var(var).is_data:
        #  if not train_program.blocks[0].var(var).is_data:
        train_program.blocks[0].var(var).persistable = True
        print(var, train_program.blocks[0].var(var).persistable, train_program.blocks[0].var(var).shape)

    # initialize
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    print('startup_program', fluid.default_startup_program())
    print('train_program', train_program)
    #  print('test_program', test_program)

    if debug:
        var_name_list = ("cos_sim_1.tmp_0@GRAD", "fc_2.tmp_1@GRAD", "fc_2.tmp_0@GRAD", "softsign_2.tmp_0@GRAD", "reduce_sum_2.tmp_0@GRAD", "stack_2.tmp_0@GRAD", "sequence_pool_23.tmp_0@GRAD", "sequence_pool_23.tmp_0@GRAD", "embedding_23.tmp_0@GRAD", "PyramidHash_emb_0@GRAD@RENAME@0", "PyramidHash_emb_0@GRAD@RENAME@1", "PyramidHash_emb_0@GRAD@RENAME@2", "PyramidHash_emb_0@GRAD@RENAME@3", "PairwiseMarginLoss_0.tmp_0@GRAD", "cos_sim_1.tmp_0", "cos_sim_1.tmp_0@GRAD", "fc_2.tmp_1@GRAD", "fc_2.tmp_0@GRAD", "softsign_2.tmp_0@GRAD", "reduce_sum_2.tmp_0@GRAD", "stack_2.tmp_0@GRAD", "sequence_pool_23.tmp_0@GRAD", "embedding_23.tmp_0@GRAD", "PyramidHash_emb_0@GRAD", "FC_1@GRAD", "EmbeddingWithVSum_emb_0@GRAD", "fc_0.w_0@GRAD", "PairwiseMarginLoss_0.tmp_0", "PairwiseMarginLoss_0.tmp_1")
        #  var_name_list = ("sequence_pool_23.tmp_0@GRAD", "embedding_23.tmp_0@GRAD", "PyramidHash_emb_0@GRAD@RENAME@0", "PyramidHash_emb_0@GRAD", "FC_1@GRAD", "EmbeddingWithVSum_emb_0@GRAD", "fc_0.w_0@GRAD", "PairwiseMarginLoss_0.tmp_0", "PairwiseMarginLoss_0.tmp_1")
        for name in var_name_list:
            train_program.blocks[0].var(name).persistable = True
            print('find var', name, train_program.blocks[0].var(name).persistable)

    # PE
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.use_cuda = use_cuda
    exec_strategy.allow_op_delay = True
    exec_strategy.num_threads = 1
    #  exec_strategy.num_threads = int(os.environ.get('THREAD_NUM', 1)) * cpu_num - 1
    #  exec_strategy.num_threads = 25
    exec_strategy.use_experimental_executor = True
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce
    #  build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
    #  build_strategy.optimize_strategy = fluid.BuildStrategy.OptimizeStrategy.NoLock
    #  pass_builder = build_strategy._create_passes_from_strategy()
    #  pass_builder.insert_pass(0, "lock_free_optimize_pass")
    train_exe = fluid.ParallelExecutor(
            use_cuda=use_cuda,
            loss_name=loss.name,
            main_program=train_program,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    test_exe = fluid.ParallelExecutor(
            use_cuda=use_cuda,
            main_program=test_program,
            share_vars_from=train_exe,
            )

    # DataFeeder
    feed_var_names = ['query_basic', 'query_phrase', 'pos_title_basic', 'pos_title_phrase', 'neg_title_basic', 'neg_title_phrase', 'label']
    feed_list = [train_program.global_block().var(var_name) for var_name in feed_var_names]
    feeder = fluid.DataFeeder(feed_list, place)
    #  batch_train_reader = feeder.decorate_reader(
            #  paddle.batch(reader.train_reader, batch_size=batch_size // cpu_num),
            #  multi_devices=true)
    batch_train_reader = feeder.decorate_reader(
            paddle.batch(reader.train_reader, batch_size=128),
            multi_devices=True)

    test_feed_var_names = ['query_basic', 'query_phrase', 'pos_title_basic', 'pos_title_phrase', 'neg_title_basic', 'neg_title_phrase']
    test_feed_list = [train_program.global_block().var(var_name) for var_name in test_feed_var_names]
    test_feeder = fluid.DataFeeder(test_feed_list, place)

    # train
    for epoch in six.moves.xrange(num_pass):
        count = 0
        total_loss = .0
        total_time = .0

        read_data_start = time.time()
        for train_data in batch_train_reader():
            read_data_end = time.time()
            #  print('read data: ', read_data_end - read_data_start)

            if count == 1 and epoch >= 1:
            #  if count % eval_iter == 0:
                print('start eval')
                t2 = time.time()
                #  with open('./eval_log/train_mini_data_' + str(epoch) + '_' + str(count) + '_' + str(time.time()), 'w') as f:
                with open('./eval_res/z_' + paddle.version.commit + 'sgd_nolock_result_' + str(epoch) + '_' + str(time.time()), 'w') as f:
                    test_batch_reader = paddle.batch(
                            reader.test_reader,
                            #  batch_size=cpu_num * 128)
                            batch_size=1280)
                    for test_data in test_batch_reader():
                        qids = []
                        labels = []
                        data_list = []
                        for one_data in test_data:
                            qids.append(one_data[0])
                            labels.append(int(one_data[-1][0]))
                            data_list.append((one_data[1:-1]))
                        predicts = test_exe.run(feed=test_feeder.feed(data_list), fetch_list=[pos_sim.name])
                        scores = np.array(predicts[0])

                        for qid, label, score in six.moves.zip(qids, labels, scores):
                            f.write(str(qid) + '\t' + str(score[0]) + '\t' + str(label) + '\n')

                print('end eval', time.time() - t2)

                start = time.time()


            if epoch == 0 and count == 5:
                profiler.start_profiler("CPU")
            elif epoch == 0 and count == 10:
                profiler.stop_profiler("total", "/paddle/Pyramid_DNN/fluid/profile")

            t1 = time.time()

            cost = train_exe.run(feed=train_data, fetch_list=[loss.name])

            total_time += time.time() - t1
            total_loss += np.array(cost[0]).mean()
            count += 1

            if debug:
                for name in var_name_list:
                    var = np.array(fluid.executor._fetch_var(name, return_numpy=False))
                    if name == "PyramidHash_emb_0@GRAD@RENAME@0":
                        print('fetch var', name, var)
                        print('check not zero', name, np.count_nonzero(var))

                    print('fetch var', name, var)
                    print('check nan var', name, np.isnan(var).any())
                    print('check inf var', name, np.isinf(var).any())

            if count % print_iter == 0:
                print('epoch: %d, batch_id: %d, avg_cost: %s, avg_time: %f' % (epoch, count, total_loss / print_iter, float(total_time) / print_iter))
                import sys
                sys.stdout.flush()
                total_time = .0
                total_loss = .0

            read_data_start = time.time()

        #  end = time.time()
        #  print('train time: ', end - start)

        #  end = time.time()
        #  print('epoch time: ', end - start)



if __name__ == '__main__':
    train()

