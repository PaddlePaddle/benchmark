# PaddlePaddle OP Benchmark

## 目录
* [功能介绍](#功能介绍)
* [测试要素](#测试要素)
* [使用方法](#使用方法)

## 功能介绍
- OP Benchmark系统的主要功能如下:
  - 测试OP的性能和精度, 验证OP优化效果
  - 拦截OP逻辑被修改而导致性能下降的问题

## 测试要素
- OP Benchmark系统依赖以下两方面完成OP测试:
  - 测试脚本：以Python格式编写的API测试脚本，分为动态图脚本和静态图脚本，分别位于[tests](https://github.com/PaddlePaddle/benchmark/tree/master/api/tests)和[test_v2](https://github.com/PaddlePaddle/benchmark/tree/master/api/tests_v2)(静态图测试脚本未来会统一到`tests`模块)两个目录中.
  - 配置信息：以json格式存储的被测参数信息, 主要位于[configs](https://github.com/PaddlePaddle/benchmark/tree/master/api/tests_v2/configs)目录中


## 使用方法
- 单任务测试
  - 性能测试：OP Benchmark支持OP精度和性能测试，默认测试飞桨框架OP前向+反向性能。用户进入测试脚本目录，输入以下指令即可执行测试任务。另外，用户只需要替换`api_name`和`config_id`就能执行不同的测试任务。
      ```shell
      bash run.sh api_name config_id

      # api_name ：测试脚本的文件名
      # config_id：配置信息在json中的id号
      ```

  - 精度测试：若想测试OP的性能，仅需要在上述指令后加入`accuracy`关键字即可
      ```shell
      bash run.sh api_name config_id accuracy
      ```

- 多任务测试
  - OP Benchmark支持一次性完成多OP、多配置的测试任务。用户需进入[deploy](https://github.com/PaddlePaddle/benchmark/tree/master/api/deploy)目录，仿照[api_info_v2](https://github.com/PaddlePaddle/benchmark/blob/master/api/deploy/api_info_v2.txt)文件构建自己需要测试的OP列表，再输入以下指令，即可全量测试这些OP的在全部配置信息下的性能或精度。
    ```shell
      bash main_control.sh test_dir config_dir result_dir gpu_id device_set task_set list_file framework testing_mode op_name precision

      # test_dir       : 被测OP的脚本目录，可设置tests、tests_v2
      # config_dir     : 被测OP的配置目录，可设置tests_v2/configs，也可以设置为用户自己指定的OP配置目录
      # result_dir     : 测试结果存放的目录
      # gpu_id         : 测试使用的GPU标号，可指定多个GPU并行测试，如"0,1"
      # device_set     : 测试设备，可设置gpu、cpu、both，默认值为both
      # task_set       : 测试任务，可设置speed、accuracy、both，默认值为both
      # list_file      : 测试op的信息列表, 如: api_info_v2.txt，默认由collect_api_info.py脚本自动生成
      # framework      : 测试框架，如paddle、tensorflow、pytorch、both，默认值为both
      # testing_mode   : 测试动态图dynamic或静态图static，默认值为dynamic
      # op_name        : 测试的OP，默认值为None，即测试全量算子
      # precision      : 测试的精度，可设置fp32、fp16、both，默认值为fp32
    ```
  - OP Benchmark还提供了多任务测试结果的汇总功能，用户在[deploy](https://github.com/PaddlePaddle/benchmark/tree/master/api/deploy)使用`python summary.py -h` 可以得到查看使用帮助，此外，在该目录下还可以使用如下命令生成数据的汇总信息.
    ```shell
      python summary.py result_dir --dump_to_excel True

      # result_dir    : 测试结果存放的目录
      # ------------------------------------------------------------
      # dump_to_excel : 汇总数据到 excel 表格   [True|default = False]
      # dump_to_text  : 汇总数据到 text 文件    [True|default = False]
      # dump_to_json  : 汇总数据到 json 文件    [True|default = False]
      # dump_to_mysql : 汇总数据到 mysql 数据库 [True|default = False]
    ```
