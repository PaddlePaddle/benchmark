from common_import import *


class Conv2dConfig(APIConfig):
    def __init__(self, op_type="conv2d"):
        super(Conv2dConfig, self).__init__(op_type)
        self.feed_spec = [
            {
                "range": [-1, 1]
            },  # input
            {
                "range": [-1, 1],
            }  # filters
        ]

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(Conv2dConfig, self).init_from_json(filename, config_id,
                                                 unknown_dim)
        if not use_gpu() and self.data_format == "NCHW":
            print(
                "Warning:\n"
                "  1. tf is disabled because the tf's conv ops currently only "
                "supports the NHWC tensor format on the CPU. Please add a rule "
                "to support it.\n")
            self.run_tf = False

        if isinstance(self.padding, int):
            self.padding = [self.padding, self.padding]
        if self.data_format == "NCHW":
            self.num_channels = self.x_shape[1]
        elif self.data_format == "NHWC":
            self.num_channels = self.x_shape[3]
        if self.groups is None:
            self.groups = 1
        if self.num_channels % self.groups != 0:
            raise ValueError(
                "the channel of input must be divisible by groups,"
                "received: the channel of input is {}, the shape of input is {}"
                ", the groups is {}".format(self.num_channels, self.x_shape,
                                            self.groups))


class PaddleConv2d(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight',
            shape=config.weight_shape,
            dtype=config.weight_dtype)
        self.feed_list = [x, weight]

    def run_graph(self, config):
        result = paddle.nn.functional.conv2d(
            x=self.feed_list[0],
            weight=self.feed_list[1],
            bias=None,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
            groups=config.groups,
            data_format=config.data_format)

        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, self.feed_list)


class TorchConv2d(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight', shape=config.weight_shape, dtype=config.x_dtype)
        self.feed_list = [x, weight]

    def run_graph(self, config):
        result = torch.nn.functional.conv2d(
            input=self.feed_list[0],
            weight=self.feed_list[1],
            bias=None,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
            groups=config.groups)

        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, self.feed_list)


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddleConv2d(),
        torch_obj=TorchConv2d(),
        config=Conv2dConfig())
