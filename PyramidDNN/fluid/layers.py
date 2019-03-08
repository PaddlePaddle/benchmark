from __future__ import print_function

import paddle
import paddle.fluid as fluid

__all__ = ['reshape', 'embedding']

def reshape(x, shape, actual_shape=None, act=None, inplace=True, name=None):
    """
    Gives a new shape to the input Tensor without changing its data.

    The target shape can be given by :attr:`shape` or :attr:`actual_shape`.
    :attr:`shape` is a list of integer while :attr:`actual_shape` is a tensor
    variable. :attr:`actual_shape` has a higher priority than :attr:`shape`
    if it is provided, while :attr:`shape` still should be set correctly to
    gurantee shape inference in compile-time.

    Some tricks exist when specifying the target shape.

    1. -1 means the value of this dimension is inferred from the total element
    number of x and remaining dimensions. Thus one and only one dimension can
    be set -1.

    2. 0 means the actual dimension value is going to be copied from the
    corresponding dimension of x. The indice of 0s in shape can not exceed
    Rank(X).

    Here are some examples to explain it.

    1. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    is [6, 8], the reshape operator will transform x into a 2-D tensor with
    shape [6, 8] and leaving x's data unchanged.

    2. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    specified is [2, 3, -1, 2], the reshape operator will transform x into a
    4-D tensor with shape [2, 3, 4, 2] and leaving x's data unchanged. In this
    case, one dimension of the target shape is set to -1, the value of this
    dimension is inferred from the total element number of x and remaining
    dimensions.

    3. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    is [-1, 0, 3, 2], the reshape operator will transform x into a 4-D tensor
    with shape [2, 4, 3, 2] and leaving x's data unchanged. In this case,
    besides -1, 0 means the actual dimension value is going to be copied from
    the corresponding dimension of x.

    Args:
        x(variable): The input tensor.
        shape(list): The new shape. At most one dimension of the new shape can
                     be -1.
        actual_shape(variable): An optional input. If provided, reshape
                                according to this given shape rather than
                                :attr:`shape` specifying shape. That is to
                                say :attr:`actual_shape` has a higher priority
                                than :attr:`shape`.
        act (str): The non-linear activation to be applied to output variable.
        inplace(bool): If this flag is set true, the output
                       shares data with input without copying, otherwise
                       a new output tensor is created
                       whose data is copied from input x.
        name (str): The name of this layer. It is optional.

    Returns:
        Variable: The output tensor.

    Raises:
        TypeError: if actual_shape is neither Variable nor None.

    Examples:
        .. code-block:: python

            data = fluid.layers.data(
                name='data', shape=[2, 4, 6], dtype='float32')
            reshaped = fluid.layers.reshape(
                x=data, shape=[-1, 0, 3, 2], act='tanh', inplace=True)
    """

    if not (isinstance(shape, list) or isinstance(shape, tuple)):
        raise ValueError("Input shape must be a python list or tuple.")
    inputs = {"X": x}
    if isinstance(actual_shape, fluid.framework.Variable):
        inputs["Shape"] = actual_shape
    elif actual_shape is not None:
        raise TypeError("actual_shape should either be Variable or None")

    # Validate the shape
    unk_dim_idx = -1
    for dim_idx, dim_size in enumerate(shape):
        if dim_size == -1:
            assert unk_dim_idx == -1, (
                "Only one dimension in shape can be unknown.")
            unk_dim_idx = dim_idx
        elif dim_size == 0:
            assert dim_idx < len(x.shape), (
                "The indice of 0s in shape can not exceed Rank(X).")
        else:
            assert dim_size > 0, (
                "Each dimension size given in shape must not be negtive "
                "except one unknown dimension.")

    helper = fluid.layer_helper.LayerHelper("reshape2", **locals())
    out = x if inplace else helper.create_variable_for_type_inference(dtype=x.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="reshape2",
        inputs=inputs,
        attrs={"shape": shape},
        outputs={"Out": out,
                 "XShape": x_shape})

    return helper.append_activation(out)


def embedding(input,
              size,
              is_sparse=False,
              is_distributed=False,
              padding_idx=None,
              param_attr=None,
              dtype='float32'):
    """
    **Embedding Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    a lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    All the input variables are passed in as local variables to the LayerHelper
    constructor.

    Args:
        input(Variable): The tensor variable containing the IDs.
        size(tuple|list): The shape of the look up table parameter. It should
            have two elements which indicate the size of the dictionary of
            embeddings and the size of each embedding vector respectively.
        is_sparse(bool): The flag indicating whether to use sparse update.
        is_distributed(bool): Whether to run lookup table from remote parameter server.
        padding_idx(int|long|None): If :attr:`None`, it makes no effect to lookup.
            Otherwise the given :attr:`padding_idx` indicates padding the output
            with zeros whenever lookup encounters it in :attr:`input`. If
            :math:`padding_idx < 0`, the :attr:`padding_idx` to use in lookup is
            :math:`size[0] + dim`.
        param_attr(ParamAttr): Parameters for this layer
        dtype(np.dtype|core.VarDesc.VarType|str): The type of data : float32, float_16, int etc

    Returns:
        Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          dict_size = len(dataset.ids)
          data = fluid.layers.data(name='ids', shape=[32, 32], dtype='float32')
          fc = fluid.layers.embedding(input=data, size=[dict_size, 16])
    """

    helper = fluid.layer_helper.LayerHelper('embedding', **locals())
    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=dtype, is_bias=False)
    tmp = helper.create_variable_for_type_inference(dtype)
    padding_idx = -1 if padding_idx is None else padding_idx if padding_idx >= 0 else (
        size[0] + padding_idx)
    helper.append_op(
        type='lookup_table',
        inputs={'Ids': input,
                'W': w},
        outputs={'Out': tmp},
        attrs={
            'grad_inplace': True,
            'is_sparse': is_sparse,
            'is_distributed': is_distributed,
            'padding_idx': padding_idx
        })
    return tmp


def fused_embedding_seq_pool(input,
              size,
              is_sparse=False,
              is_distributed=False,
              padding_idx=None,
              param_attr=None,
              dtype='float32'):
    """
    **Embedding Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    a lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    All the input variables are passed in as local variables to the LayerHelper
    constructor.

    Args:
        input(Variable): The tensor variable containing the IDs.
        size(tuple|list): The shape of the look up table parameter. It should
            have two elements which indicate the size of the dictionary of
            embeddings and the size of each embedding vector respectively.
        is_sparse(bool): The flag indicating whether to use sparse update.
        is_distributed(bool): Whether to run lookup table from remote parameter server.
        padding_idx(int|long|None): If :attr:`None`, it makes no effect to lookup.
            Otherwise the given :attr:`padding_idx` indicates padding the output
            with zeros whenever lookup encounters it in :attr:`input`. If
            :math:`padding_idx < 0`, the :attr:`padding_idx` to use in lookup is
            :math:`size[0] + dim`.
        param_attr(ParamAttr): Parameters for this layer
        dtype(np.dtype|core.VarDesc.VarType|str): The type of data : float32, float_16, int etc

    Returns:
        Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          dict_size = len(dataset.ids)
          data = fluid.layers.data(name='ids', shape=[32, 32], dtype='float32')
          fc = fluid.layers.embedding(input=data, size=[dict_size, 16])
    """

    helper = fluid.layer_helper.LayerHelper('fused_embedding_seq_pool', **locals())
    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=dtype, is_bias=False)
    tmp = helper.create_variable_for_type_inference(dtype)
    padding_idx = -1 if padding_idx is None else padding_idx if padding_idx >= 0 else (
        size[0] + padding_idx)
    helper.append_op(
        type='fused_embedding_seq_pool',
        inputs={'Ids': input,
                'W': w},
        outputs={'Out': tmp},
        attrs={
            'is_sparse': is_sparse,
            'combiner': 'sum'
        })
    return tmp


def fused_hash_embedding_seq_pool(x,
              size,
              hash_size,
              num_hash,
              is_sparse=False,
              is_distributed=False,
              padding_idx=None,
              param_attr=None,
              dtype='float32'):
    """
    **Fused Hash, Embedding and Sequence Pool Op Layer**
    """

    helper = fluid.layer_helper.LayerHelper('fused_hash_embedding_seq_pool', **locals())
    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=dtype, is_bias=False)
    tmp = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='fused_hash_embedding_seq_pool',
        inputs={'X': x,
                'W': w},
        outputs={'Out': tmp},
        attrs={
            'num_hash': num_hash,
            'mod_by': hash_size,
            'is_sparse': is_sparse,
            'combiner': 'sum'
        })
    return tmp
