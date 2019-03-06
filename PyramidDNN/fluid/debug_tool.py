import numpy as np

names = []
grad_names = []
import collections

name_dict = collections.OrderedDict()
name_dict['embedding_para'] = 1
#'''
name_dict['lstmp_0.b_0'] = 21
name_dict['fw_layer1_gate_w'] = 22
name_dict['lstmp_0.w_0'] = 22
name_dict['lstmp_0.w_1'] = 23

name_dict['lstmp_1.b_0'] = 31
name_dict['fw_layer2_gate_w'] = 32
name_dict['lstmp_1.w_0'] = 32
name_dict['lstmp_1.w_1'] = 33

name_dict['lstmp_2.b_0'] = 41
name_dict['bw_layer1_gate_w'] = 42
name_dict['lstmp_2.w_0'] = 42
name_dict['lstmp_2.w_1'] = 43

name_dict['lstmp_3.b_0'] = 51
name_dict['bw_layer2_gate_w'] = 52
name_dict['lstmp_3.w_0'] = 52
name_dict['lstmp_3.w_1'] = 53
#'''
name_dict['softmax_weight'] = 62
name_dict['softmax_bias'] = 61

slot_dict = {}


def init_slot():
    global slot_dict
    slot_dict = {}


def name2slot(para_name, exact=False):
    res = []
    if exact:
        if para_name in name_dict:
            return [name_dict[para_name]]
        else:
            return []
    for key_name in name_dict.keys():
        if para_name.find(key_name) >= 0:
            res.append(name_dict[key_name])
    return res


def update_slot(slots, p_array):
    p_mean, p_max, p_min, p_num = p_array.mean(), p_array.max(), p_array.min(
    ), np.prod(p_array.shape)
    for slot in slots:
        if slot in slot_dict:
            s_mean, s_max, s_min, s_num = slot_dict[slot]
            s_mean = (s_mean * s_num + p_mean * p_num) / (p_num + s_num)
            s_max = max(s_max, p_max)
            s_min = min(s_min, p_min)
            s_num = p_num + s_num
            slot_dict[slot] = [s_mean, s_max, s_min, s_num]
        else:
            slot_dict[slot] = [p_mean, p_max, p_min, p_num]


def record_slot(logger):
    for slot in slot_dict:
        logger.info("slot:" + "\t".join(
            [str(round(x, 10)) for x in [slot] + slot_dict[slot]]))


def var_print(tag, p_array, p_name, name, detail, logger):
    param_num = np.prod(p_array.shape)
    p_array3 = np.multiply(np.multiply(p_array, p_array), p_array)
    logger.info(
        tag +
        ": {0} ({1}),  l3={2} sum={3}  max={4}  min={5} mean={6} num={7} {8}".
        format(p_name, name,
               p_array3.sum(),
               p_array.sum(),
               p_array.max(),
               p_array.min(), p_array.mean(), p_array.shape, param_num))
    if detail:
        logger.info(" ".join([
            tag + "[", p_name, '] shape [', str(p_array.shape), ']', str(
                p_array)
        ]))


def save_var(p_array, name, logger, args):
    if args.save_para_path:
        if name2slot(name, exact=True):
            name = 'slot_' + str(name2slot(name, exact=True)[0])
        else:
            name = name.replace('/', '%')
        with open(os.path.join(args.save_para_path, name + '.data'),
                  'wb') as fout:
            pickle.dump(p_array, fout)


def save_para(train_prog, train_exe, logger, args=None):
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]
    for var in param_list:
        p_name = var.name
        p_array = np.array(train_exe.scope.find_var(p_name).get_tensor(
        )).astype('float64')
        save_var(p_array, p_name, logger, args)


def load_var(tensor, slot, place, logger, args):
    with open(
            os.path.join(args.para_load_dir, 'slot_' + str(slot[0]) + '.data'),
            'rb') as fin:
        p_array = pickle.load(fin)
        if slot in [22, 32, 42, 52]:
            tensor.set(p_array.astype(np.float32), place)
        else:
            tensor.set(p_array.astype(np.float32), place)


def listDir(rootDir):
    res = []
    for filename in os.listdir(rootDir):
        pathname = os.path.join(rootDir, filename)
        if (os.path.isfile(pathname)):
            res.append(pathname)
    return res


#load from slot file
def load_params(train_prog, train_exe, place, logger, args=None):
    if not args.para_load_dir:
        return
    logger.info('loading para from {}'.format(args.para_load_dir))
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]
    for data in listDir(args.para_load_dir):
        slot = int(data.split('_')[1].split('.')[0])
        with open(data, 'rb') as fin:
            if six.PY2:
                p_array = pickle.load(fin)
            else:
                p_array = pickle.load(fin, encoding='bytes')
            p_array = p_array.reshape((-1))
            offset = 0
            for name in name_dict:
                s = name_dict[name]
                if s == slot:
                    card = 0
                    #for scope in [train_exe.scope]:#train_exe.executor.local_scopes():
                    for scope in train_exe.executor.local_scopes():
                        tensor = scope.find_var(name).get_tensor()
                        shape = tensor.shape()
                        tensor_len = np.prod(shape)
                        new_array = p_array[offset:offset + tensor_len]
                        new_array = new_array.reshape(shape)
                        if args.use_gpu:
                            placex = fluid.CUDAPlace(card)
                        else:
                            placex = fluid.CPUPlace()
                        tensor.set(new_array.astype(np.float32), placex)
                        logger.info('card {} loaded {}[{}] from {}[{}:{}]'.
                                    format(card, name, shape, data, offset,
                                           offset + tensor_len))
                        card = card + 1
                    offset += tensor_len


def load_para(train_prog, train_exe, place, logger, args=None):
    if not args.para_load_dir:
        return
    logger.info('loading para form {}'.format(args.para_load_dir))
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]
    for var in param_list:
        p_name = var.name
        tensor = train_exe.scope.find_var(p_name).get_tensor()
        if name2slot(var.name, exact=True):
            slot = name2slot(var.name, exact=True)
            load_var(tensor, slot, place, logger, args)


names = []
grad_names = [
]  #[['create_parameter_0.w_0@GRAD', 'create_parameter_0.w_0']]#,['embedding_para@GRAD', 'embedding_para']]


def debug_init(train_prog, vars, vars_name):
    for i in range(len(vars)):
        name = vars[i].name + '@GRAD'
        grad_names.append([name, vars_name[i]])
        name = vars[i].name
        names.append([name, vars_name[i]])
    for name in names:
        train_prog.block(0).var(name[0]).persistable = True
    for name in grad_names:
        if train_prog.block(0).has_var(name[0]):
            train_prog.block(0).var(name[0]).persistable = True

    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]
    for p_name in param_name_list:
        p_name = p_name + '@GRAD'
        if train_prog.block(0).has_var(p_name):
            train_prog.block(0).var(p_name).persistable = True


def debug_print(scope, logger, args):
    if not args.para_print:
        return
    for name_pair in names:
        name = name_pair[0]
        p_name = name_pair[1]
        if not scope.find_var(name):
            logger.info("var: {0} not find".format(p_name))
            continue
        p_array = np.array(scope.find_var(name).get_tensor()).astype('float64')
        var_print('var', p_array, p_name, name, args.detail, logger)
    for name_pair in grad_names:
        name = name_pair[0]
        p_name = name_pair[1]
        if not scope.find_var(name):
            logger.info("grad: {0} not find".format(p_name))
            continue
        p_array = np.array(scope.find_var(name).get_tensor()).astype('float64')
        var_print('grad', p_array, p_name, name, args.detail, logger)


def vars_print(logger, args, vars=None, grad_vars=None):
    if not args.para_print:
        return
    for var, vname in zip(vars):
        name, p_name = vname
        p_array = np.array(var).astype('float64')
        var_print('var', p_array, p_name, name, args.detail, logger)
    for grad, gname in zip(grad_vars):
        name, p_name = gname
        p_array = np.array(grad).astype('float64')
        var_print('grad', p_array, p_name, name, args.detail, logger)


def print_para(train_prog, train_exe, logger, optimizer=None, args=None):
    if not args.para_print:
        return
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]
    card = 0
    for scope in train_exe.executor.local_scopes():
        init_slot()
        num_sum = 0
        logger.info('card {}'.format(card))
        debug_print(scope, logger, args)
        for p_name in param_name_list:
            p_name = p_name + '@GRAD'
            if not scope.find_var(p_name):
                logger.info("grad para: {0} not find".format(p_name))
                #import pdb; pdb.set_trace()
                continue
            try:
                p_array = np.array(scope.find_var(p_name).get_tensor())
            except:
                #import pdb; pdb.set_trace()
                logger.info("grad para: {0} failed".format(p_name))
                continue
            param_num = np.prod(p_array.shape)
            var_print('grad para', p_array, p_name, p_name, args.detail,
                      logger)
        if optimizer:
            for p_name in param_name_list:
                acc_str = 'moment'
                try:
                    acc = optimizer._accumulators[acc_str][p_name]
                    p_array = np.array(scope.find_var(acc.name).get_tensor())
                    var_print(acc_str, p_array, p_name, acc.name, args.detail,
                          logger)
                except:
                    logger.info("grad monent para: {0} failed".format(p_name))
        for p_name in param_name_list:
            p_array = np.array(scope.find_var(p_name).get_tensor())
            slots = name2slot(p_name)
            if slots:
                update_slot(slots, p_array)
            param_num = np.prod(p_array.shape)
            num_sum = num_sum + param_num
            var_print('para', p_array, p_name, p_name, args.detail, logger)
        record_slot(logger)
        logger.info("total param num: {0}".format(num_sum))

        card = card + 1
