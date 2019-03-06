from __future__ import print_function

import math
import six
import paddle.fluid as fluid
import layers

def test_net(dict_size, hash_size):
    qb = fluid.layers.data(name='query_basic', shape=[1], dtype="int64", lod_level=1)

    qp = fluid.layers.data(name='query_phrase', shape=[1], dtype="int64", lod_level=1)

    p_tb = fluid.layers.data(name='pos_title_basic', shape=[1], dtype="int64", lod_level=1)

    p_tp = fluid.layers.data(name='pos_title_phrase', shape=[1], dtype="int64", lod_level=1)

    # query fc output
    q_fc = from_emb_to_fc(qb, qp, dict_size=dict_size, hash_size=hash_size, share_fc=False)

    # positive title fc output
    pt_fc = from_emb_to_fc(p_tb, p_tp, dict_size=dict_size, hash_size=hash_size, share_fc=True)

    ps = fluid.layers.cos_sim(q_fc, pt_fc)

    return ps

"""
Args:
    dict_size: The size of word dict.
    hash_size: The space size of hash operation.
    emb_size: The embedding size of word. Default 256.
    is_train: If the network is contructed for training. Default True.

Return:
    Program, Reader
"""
def net(dict_size, hash_size, emb_size=256, is_train=True):
    # embedding could only accept int64 type as ids
    label = fluid.layers.data(name='label', shape=[1], dtype='float32')

    qb = fluid.layers.data(name='query_basic', shape=[1], dtype="int64", lod_level=1)

    qp = fluid.layers.data(name='query_phrase', shape=[1], dtype="int64", lod_level=1)

    p_tb = fluid.layers.data(name='pos_title_basic', shape=[1], dtype="int64", lod_level=1)

    p_tp = fluid.layers.data(name='pos_title_phrase', shape=[1], dtype="int64", lod_level=1)

    n_tb = fluid.layers.data(name='neg_title_basic', shape=[1], dtype="int64", lod_level=1)

    n_tp = fluid.layers.data(name='neg_title_phrase', shape=[1], dtype="int64", lod_level=1)

    # query fc output
    q_fc = from_emb_to_fc(qb, qp, dict_size=dict_size, hash_size=hash_size, share_fc=False)

    # positive title fc output
    pt_fc = from_emb_to_fc(p_tb, p_tp, dict_size=dict_size, hash_size=hash_size, share_fc=True)

    # negative title fc output
    nt_fc = from_emb_to_fc(n_tb, n_tp, dict_size=dict_size, hash_size=hash_size, share_fc=True)

    ps = fluid.layers.cos_sim(q_fc, pt_fc)

    train_program = fluid.default_main_program()

    test_program = train_program.clone(for_test=True)

    ns = fluid.layers.cos_sim(q_fc, nt_fc)

    loss = fluid.layers.margin_rank_loss(label, ps, ns, name="PairwiseMarginLoss_0", margin=0.1)

    mean_loss = fluid.layers.reduce_mean(loss, dim=0)

    return mean_loss, ps, train_program, test_program

def from_emb_to_fc(qb, qp, hash_size, dict_size, share_fc):
    # query basic
    qb_ss = emb_layer(qb, dict_size=dict_size, emb_size=256)

    #  qb_hashed = pyramid_hash_layer(qb, num_hash=16, hash_size=hash_size, emb_size=256, dict_size=dict_size)
    qb_hashed = fused_pyramid_hash_layer(qb, num_hash=16, hash_size=hash_size, emb_size=256, dict_size=dict_size)

    # query phrase
    qp_ss = emb_layer(qp, dict_size=dict_size, emb_size=256)

    #  qp_hashed = pyramid_hash_layer(qp, num_hash=16, hash_size=hash_size, emb_size=256, dict_size=dict_size)
    qp_hashed = fused_pyramid_hash_layer(qp, num_hash=16, hash_size=hash_size, emb_size=256, dict_size=dict_size)

    print('qb_ss.shape', qb_ss.shape)

    query_emb_list = []
    query_emb_list.append(qb_ss)
    query_emb_list.append(qp_ss)
    query_emb_list.extend(qb_hashed)
    query_emb_list.extend(qp_hashed)

    #  query_emb_sum_list = []
    #  for query_emb in query_emb_list:
        #  query_emb_sum_list.append(vsum_layer(x=query_emb, dim=1))

    # concat basic and phrase
    #  stack_query = fluid.layers.stack(query_emb_list, axis=1)

    #  print('stack_query.shape', stack_query.shape)

    #  qvs = vsum_layer(x=stack_query, dim=1)

    qvs = fluid.layers.sum(query_emb_list)

    print('qvs.shape', qvs.shape)

    qss = softsign_layer(x=qvs)

    q_fc = fluid.layers.fc(
            qss,
            num_flatten_dims=1,
            size=256,
            param_attr=fluid.ParamAttr(
                name='FC_1' if share_fc else None,
                learning_rate=200,
                initializer=fluid.initializer.Uniform(low=-math.sqrt(6.0 / (256 * 256)), high=math.sqrt(6.0 / (256 * 256)), seed=0)),
            bias_attr=fluid.ParamAttr(
                name='FC_b_1' if share_fc else None,
                learning_rate=200,
                initializer=fluid.initializer.Uniform(low=-math.sqrt(6.0 / (256 * 256)), high=math.sqrt(6.0 / (256 * 256)), seed=0)))

    return q_fc

def softsign_layer(x, shared=True):
    softsign = fluid.layers.softsign(x=x)

    return softsign

def vsum_layer(x, dim=2, shared=True):
    reduce_sum = fluid.layers.reduce_sum(input=x, dim=dim)

    return reduce_sum

def emb_layer(x, dict_size, emb_size=256, shared=True):
    print("emb_layer input shape ", x.shape)
    pool = layers.fused_embedding_seq_pool(
            input=x, size=[dict_size, emb_size], is_sparse=True,
            param_attr=fluid.ParamAttr(
                name='EmbeddingWithVSum_emb_0',
                learning_rate=640,
                initializer=fluid.initializer.Uniform(low=-math.sqrt(6.0 / (emb_size * dict_size)), high=math.sqrt(6.0 / (emb_size * dict_size)), seed=0)) if shared else None,
            dtype='float32')
    #  emb = layers.embedding(
            #  input=x, size=[dict_size, emb_size], is_sparse=True,
            #  param_attr=fluid.ParamAttr(
                #  name='EmbeddingWithVSum_emb_0',
                #  learning_rate=640,
                #  initializer=fluid.initializer.Uniform(low=-math.sqrt(6.0 / (emb_size * dict_size)), high=math.sqrt(6.0 / (emb_size * dict_size)), seed=0)) if shared else None,
            #  dtype='float32')

    #  pool = fluid.layers.sequence_pool(input=emb, pool_type='sum')

    return pool

def pyramid_hash_layer(x, hash_size, dict_size, num_hash=16, emb_size=256, min_win_size=2, max_win_size=4, shared=True):
    emb_list = []
    for win_size in six.moves.xrange(min_win_size, max_win_size + 1):
        print("pyramid_hash_layer input shape ", x.shape)
        seq_enum = fluid.layers.sequence_enumerate(x, win_size)
        print("seq_enum.shape ", seq_enum.shape)
        xxhash = fluid.layers.hash(seq_enum, hash_size=hash_size, num_hash=num_hash)
        print("xxhash.shape ", xxhash.shape)

        pool = layers.fused_embedding_seq_pool(
            input=xxhash, size=[hash_size, emb_size // num_hash], is_sparse=True,
            param_attr=fluid.ParamAttr(
                name='PyramidHash_emb_0',
                learning_rate=640,
                initializer=fluid.initializer.Uniform(low=-math.sqrt(6.0 / (emb_size * dict_size)), high=math.sqrt(6.0 / (emb_size * dict_size)), seed=0)) if shared else None,
            dtype='float32')
        #  emb = layers.embedding(
            #  input=xxhash, size=[hash_size, emb_size // num_hash], is_sparse=True,
            #  param_attr=fluid.ParamAttr(
                #  name='PyramidHash_emb_0',
                #  learning_rate=640,
                #  initializer=fluid.initializer.Uniform(low=-math.sqrt(6.0 / (emb_size * dict_size)), high=math.sqrt(6.0 / (emb_size * dict_size)), seed=0)) if shared else None,
            #  dtype='float32')

        #  print(emb.name, emb.shape)
        #  reshape_emb = layers.reshape(x=emb, shape=[-1, emb_size], inplace=True)

        #  pool = fluid.layers.sequence_pool(input=reshape_emb, pool_type='sum')
        emb_list.append(pool)

    return emb_list

def fused_pyramid_hash_layer(x, hash_size, dict_size, num_hash=16, emb_size=256, min_win_size=2, max_win_size=4, shared=True):
    emb_list = []
    seq_enum_list = []
    for win_size in six.moves.xrange(min_win_size, max_win_size + 1):
        print("fused_pyramid_hash_layer input shape ", x.shape)
        seq_enum = fluid.layers.sequence_enumerate(x, win_size)
        seq_enum_list.append(seq_enum)

    pool = layers.fused_hash_embedding_seq_pool(
        x=seq_enum_list, size=[hash_size, emb_size // num_hash], is_sparse=True,
        hash_size=hash_size, num_hash=num_hash,
        param_attr=fluid.ParamAttr(
            name='PyramidHash_emb_0',
            learning_rate=640,
            initializer=fluid.initializer.Uniform(low=-math.sqrt(6.0 / (emb_size * dict_size)), high=math.sqrt(6.0 / (emb_size * dict_size)), seed=0)) if shared else None,
        dtype='float32')
    emb_list.append(pool)

    return emb_list

if __name__ == '__main__':
    loss, pos_sim, train_program, test_program = net(hash_size=100000, dict_size=100000)
    print(train_program)
