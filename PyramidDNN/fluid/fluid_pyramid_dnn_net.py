from __future__ import print_function

import math
import six
import paddle.fluid as fluid
import layers

__all__ = ['net', 'test_net']

"""
Args:
    dict_size: The size of word dict.
    hash_size: The space size of hash operation.
    emb_size: The embedding size of word. Default 256.
    is_train: If the network is contructed for training. Default True.

Return:
    Program, Reader
"""
class Pyramidnn(object):
    def __init__(self, dict_size, hash_size, queue_capacity=1000, emb_size=256, is_train=True):
        self.dict_size = dict_size
        self.hash_size = hash_size
        self.queue_capacity = queue_capacity
        self.emb_size = emb_size
        self.is_train = is_train
        self.vars=[]
        self.vars_name=[]

    def var_append(self, var, var_name):
        self.vars.append(var)
        self.vars_name.append(var_name)

    def build(self):
        self.train_main_program = fluid.Program()
        self.train_startup_program = fluid.Program()

        self.train_main_program.random_seed = 1
        self.train_startup_program.random_seed = 1

        with fluid.program_guard(self.train_main_program, self.train_startup_program):
            # embedding could only accept int64 type as ids
            py_reader = fluid.layers.py_reader(
                capacity=self.queue_capacity,
                shapes=[[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]],
                lod_levels=[1, 1, 1, 1, 1, 1, 0],
                dtypes=["int64", "int64", "int64", "int64", "int64", "int64", "float32"],
                use_double_buffer=False)
            qb, qp, p_tb, p_tp, n_tb, n_tp, label = fluid.layers.read_file(py_reader)
            name=''
            self.var_append(qb, name + 'qb')
            self.var_append(qp, name + 'qp')
            self.var_append(p_tb, name + 'p_tb')
            self.var_append(p_tp, name + 'p_tp')
            self.var_append(n_tb, name + 'n_tb')
            self.var_append(n_tp, name + 'n_tp')
            self.var_append(label, name + 'label')

            # query fc output
            q_fc = self.from_emb_to_fc(qb, qp, dict_size=self.dict_size, hash_size=self.hash_size, share_fc=False, name='q:')

            # positive title fc output
            pt_fc = self.from_emb_to_fc(p_tb, p_tp, dict_size=self.dict_size, hash_size=self.hash_size, share_fc=True, name='pt:')

            # negative title fc output
            nt_fc = self.from_emb_to_fc(n_tb, n_tp, dict_size=self.dict_size, hash_size=self.hash_size, share_fc=True, name='nt:')

            ps = fluid.layers.cos_sim(q_fc, pt_fc)

            ns = fluid.layers.cos_sim(q_fc, nt_fc)

            loss = fluid.layers.margin_rank_loss(label, ps, ns, name="PairwiseMarginLoss_0", margin=0.1)

            mean_loss = fluid.layers.reduce_mean(loss, dim=0)

            self.optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
            self.optimizer.minimize(loss)

        self.var_append(ps, name + 'ps')
        self.var_append(ns, name + 'ns')
        self.var_append(loss, name + 'loss')
        self.var_append(mean_loss, name + 'mean_loss')

        return ps.name, self.train_main_program, self.train_startup_program, py_reader

    def from_emb_to_fc(self, qb, qp, hash_size, dict_size, share_fc, name=''):
        # query basic
        qb_ss = self.emb_layer(qb, dict_size=dict_size, emb_size=256)

        qb_hashed = self.pyramid_hash_layer(qb, num_hash=16, hash_size=hash_size, emb_size=256, dict_size=dict_size, name='qb_hashed:')

        # query phrase
        qp_ss = self.emb_layer(qp, dict_size=dict_size, emb_size=256)

        qp_hashed = self.pyramid_hash_layer(qp, num_hash=16, hash_size=hash_size, emb_size=256, dict_size=dict_size, name='qp_hashed:')

        query_emb_list = []
        query_emb_list.append(qb_ss)
        query_emb_list.append(qp_ss)
        query_emb_list.extend(qb_hashed)
        query_emb_list.extend(qp_hashed)

        qvs = fluid.layers.sum(query_emb_list)

        qss = fluid.layers.softsign(x=qvs)

        q_fc = fluid.layers.fc(
                qss,
                num_flatten_dims=1,
                size=256,
                param_attr=fluid.ParamAttr(
                    name='FC_title_1' if share_fc else 'FC_query_1',
                    learning_rate=200,
                    initializer=fluid.initializer.Uniform(low=-math.sqrt(6.0 / (256 * 256)), high=math.sqrt(6.0 / (256 * 256)), seed=0)),
                bias_attr=fluid.ParamAttr(
                    name='FC_title_b_1' if share_fc else 'FC_query_b_1',
                    learning_rate=200,
                    initializer=fluid.initializer.Uniform(low=-math.sqrt(6.0 / (256 * 256)), high=math.sqrt(6.0 / (256 * 256)), seed=0)))

        self.var_append(qb_ss, name + 'qb_ss')
        self.var_append(qp_ss, name + 'qp_ss')
        self.var_append(qvs, name + 'qvs')
        self.var_append(qss, name + 'qss')
        self.var_append(q_fc, name + 'q_fc')
        return q_fc

    def emb_layer(self, x, dict_size, emb_size=256, shared=True):
        pool = layers.fused_embedding_seq_pool(
                input=x, size=[dict_size, emb_size], is_sparse=True,
                param_attr=fluid.ParamAttr(
                    name='EmbeddingWithVSum_emb_0',
                    learning_rate=640,
                    initializer=fluid.initializer.Uniform(low=-math.sqrt(6.0 / (emb_size * dict_size)), high=math.sqrt(6.0 / (emb_size * dict_size)), seed=0)) if shared else None,
                dtype='float32')

        return pool

    def pyramid_hash_layer(self, x, hash_size, dict_size, num_hash=16, emb_size=256, min_win_size=2, max_win_size=4, shared=True, name=''):
        emb_list = []
        for win_size in six.moves.xrange(min_win_size, max_win_size + 1):
            seq_enum = fluid.layers.sequence_enumerate(x, win_size)
            xxhash = fluid.layers.hash(seq_enum, hash_size=hash_size * num_hash, num_hash=num_hash)

            pool = layers.fused_embedding_seq_pool(
                input=xxhash, size=[hash_size * num_hash, emb_size // num_hash], is_sparse=True,
                param_attr=fluid.ParamAttr(
                    name='PyramidHash_emb_0',
                    learning_rate=640,
                    initializer=fluid.initializer.Uniform(low=-math.sqrt(6.0 / (emb_size * dict_size)), high=math.sqrt(6.0 / (emb_size * dict_size)), seed=0)) if shared else None,
                dtype='float32')
            
            self.var_append(seq_enum, name + "seq_enum_w" + str(win_size))
            self.var_append(xxhash, name + "xxhash_w" + str(win_size))
            self.var_append(pool, name + "pool_w" + str(win_size))
            emb_list.append(pool)

        return emb_list

    def fused_pyramid_hash_layer(self, x, hash_size, dict_size, num_hash=16, emb_size=256, min_win_size=2, max_win_size=4, shared=True):
        emb_list = []
        seq_enum_list = []

        for win_size in six.moves.xrange(min_win_size, max_win_size + 1):
            seq_enum = fluid.layers.sequence_enumerate(x, win_size)
            seq_enum_list.append(seq_enum)

        pool = layers.fused_hash_embedding_seq_pool(
            x=seq_enum_list, size=[hash_size * num_hash, emb_size // num_hash], is_sparse=True,
            hash_size=hash_size, num_hash=num_hash,
            param_attr=fluid.ParamAttr(
                name='PyramidHash_emb_0',
                learning_rate=640,
                initializer=fluid.initializer.Uniform(low=-math.sqrt(6.0 / (emb_size * dict_size)), high=math.sqrt(6.0 / (emb_size * dict_size)), seed=0)) if shared else None,
            dtype='float32')
        emb_list.append(pool)

        return emb_list

    def build_test(self, dict_size, hash_size, parallelism=1, queue_capacity=5000, emb_size=256, is_train=False):
        test_main_program = fluid.Program()
        test_startup_program = fluid.Program()

        test_main_program.random_seed = 1
        test_startup_program.random_seed = 1

        with fluid.program_guard(test_main_program, test_startup_program):
            # embedding could only accept int64 type as ids
            qb = fluid.layers.data(name='query_basic', shape=[1], dtype="int64", lod_level=1)

            qp = fluid.layers.data(name='query_phrase', shape=[1], dtype="int64", lod_level=1)

            p_tb = fluid.layers.data(name='pos_title_basic', shape=[1], dtype="int64", lod_level=1)

            p_tp = fluid.layers.data(name='pos_title_phrase', shape=[1], dtype="int64", lod_level=1)

            # query fc output
            q_fc = self.from_emb_to_fc(qb, qp, dict_size=dict_size, hash_size=hash_size, share_fc=False)

            # positive title fc output
            pt_fc = self.from_emb_to_fc(p_tb, p_tp, dict_size=dict_size, hash_size=hash_size, share_fc=True)

            ps = fluid.layers.cos_sim(q_fc, pt_fc)

            print(ps.shape)

        return test_main_program.clone(for_test=True), ps



if __name__ == '__main__':
    model = Pyramidnn(hash_size=100000, dict_size=100000)
    loss, pos_sim, train_program, test_program = model.build()
    print(train_program)
