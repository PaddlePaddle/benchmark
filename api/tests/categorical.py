from common_import import *


@benchmark_registry.register("categorical")
class CategoricalConfig(APIConfig):
    def __init__(self):
        super(CategoricalConfig, self).__init__("categorical")
        self.feed_spec = {"range": [-5, -0.1]}


@benchmark_registry.register("categorical")
class PaddleCategorical(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        logits = self.variable(
            name="logits",
            shape=config.logits_shape,
            dtype=config.logits_dtype)
        result = paddle.distribution.Categorical(logits)
        counts = result.sample([100])
        self.feed_list = [logits]
        self.fetch_list = [counts]


@benchmark_registry.register("categorical")
class TorchCategorical(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        logits = self.variable(
            name="logits",
            shape=config.logits_shape,
            dtype=config.logits_dtype)
        result = torch.distributions.categorical.Categorical(
            logits=torch.tensor(logits))
        counts = result.sample([100])
        self.feed_list = [logits]
        self.fetch_list = [counts]


@benchmark_registry.register("categorical")
class TFCategoricall(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        logits = self.variable(
            name='logits',
            shape=config.logits_shape,
            dtype=config.logits_dtype)
        counts = tf.random.categorical(logits, 100)
        self.feed_list = [logits]
        self.fetch_list = [counts]
