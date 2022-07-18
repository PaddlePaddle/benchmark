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
  - 测试脚本：以Python格式编写的API测试脚本，分为动态图脚本和静态图脚本，分别位于[dynamic_tests_v2](https://github.com/PaddlePaddle/benchmark/tree/master/api/dynamic_tests_v2)和[test_v2](https://github.com/PaddlePaddle/benchmark/tree/master/api/tests_v2)两个目录中.
  - 配置信息：以Json格式存储的被测参数信息, 主要位于[configs](https://github.com/PaddlePaddle/benchmark/tree/master/api/tests_v2/configs)目录中


## 使用方法
- 单任务测试
  - 精度测试：OP Benchmark支持OP精度和性能测试，默认测试OP精度。用户进入测试脚本目录，输入以下指令即可执行测试任务。另外，用户只需要替换`api_name`和`config_id`就能执行不同的测试任务。
      ```shell
      bash run.sh   api_name  config_id
      
      # api_name ：测试脚本的文件名
      # config_id：配置信息在Json中的id号
      ```

  - 性能测试：若想测试OP的性能，仅需要在上述指令后加入`speed`关键字即可
      ```shell
      bash run.sh   api_name  config_id  speed
      ```

- 多任务测试
  - OP Benchmark支持一次性完成多OP、多配置的测试任务。用户需进入[deploy](https://github.com/PaddlePaddle/benchmark/tree/master/api/deploy)目录，仿照[api_info_v2](https://github.com/PaddlePaddle/benchmark/blob/master/api/deploy/api_info_v2.txt)文件构建自己需要测试的OP列表，再输入以下指令，即可全量测试这些OP的在全部配置信息下的性能或精度。
    ```shell
      bash  main_control.sh  test_dir  config_dir  result_dir  gpu_id  device_set  task_set  list_file  framework mode
    
      # test_dir   : 被测OP的脚本目录​
      # config_dir : 被测OP的配置目录​
      # result_dir : 测试结果存放的目录​
      # gpu_id     : 测试使用的GPU标号​
      # device_set : 选择GPU或CPU​完成测试
      # task_set   : 测试speed或accuracy​
      # list_file  : 测试多个op的列表​, 如: api_info_v2.txt
      # framework  : 测试paddle、tensorflow、pytorch​
      # mode       : 测试动态图dynamic或静态图static
    ```
  - OP Benchmark还提供了多任务测试结果的汇总功能，用户在[deploy](https://github.com/PaddlePaddle/benchmark/tree/master/api/deploy)使用`python summary.py -h` 可以得到查看使用帮助，此外，在该目录下还可以使用如下命令生成数据的汇总信息.
    ```shell
      python summary.py  result_dir  --dump_to_excel True

      # result_dir      : 测试结果存放的目录​
      # ------------------------------------------------------------
      # dump_to_excel : 汇总数据​到 excel 表格   [True|default = False]
      # dump_to_text  : 汇总数据​到 text 文件    [True|default = False]
      # dump_to_json  : 汇总数据​到 json 文件    [True|default = False]
      # dump_to_mysql : 汇总数据​到 mysql 数据库 [True|default = False]
    ```