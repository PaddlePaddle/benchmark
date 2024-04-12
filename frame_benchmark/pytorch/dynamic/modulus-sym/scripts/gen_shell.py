import os
import os.path as osp
import shutil


shell_content = [
    [
        "pushd examples/annular_ring/annular_ring",
        "python annular_ring.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/annular_ring/annular_ring_equation_instancing",
        "python annular_ring.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/annular_ring/annular_ring_gradient_enhanced",
        "python annular_ring_gradient_enhanced.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/annular_ring/annular_ring_hardBC",
        "python annular_ring_hardBC.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/annular_ring/annular_ring_parameterized",
        "python annular_ring_parameterized.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/anti_derivative",
        "python data_informed.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/anti_derivative",
        "python physics_informed.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/bracket",
        "python bracket.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/chip_2d",
        "python chip_2d_solid_fluid_heat_transfer_flow.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/chip_2d",
        "python chip_2d_solid_fluid_heat_transfer_heat.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/chip_2d",
        "python chip_2d_solid_solid_heat_transfer.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/chip_2d",
        "python chip_2d.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/cylinder",
        "python cylinder_2d.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/darcy",
        "python darcy_AFNO.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/darcy",
        "python darcy_DeepO.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/darcy",
        "python darcy_FNO_lazy.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/darcy",
        "python darcy_FNO.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/darcy",
        "python darcy_PINO.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/fuselage_panel",
        "python panel.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/helmholtz",
        "python helmholtz_hardBC.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/helmholtz",
        "python helmholtz_ntk.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/helmholtz",
        "python helmholtz.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/ldc",
        "python ldc_2d_domain_decomposition_fbpinn.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/ldc",
        "python ldc_2d_domain_decomposition.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/ldc",
        "python ldc_2d_importance_sampling.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/ldc",
        "python ldc_2d_zeroEq.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/ldc",
        "python ldc_2d.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/limerock/limerock_hFTB",
        "python limerock_flow.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/limerock/limerock_hFTB",
        "python limerock_thermal.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/limerock/limerock_transfer_learning",
        "python limerock_flow.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/ode_spring_mass",
        "python spring_mass_solver.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/seismic_wave",
        "python wave_2d.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/surface_pde/sphere",
        "python sphere.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/taylor_green",
        "python taylor_green_causal.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/taylor_green",
        "python taylor_green.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/three_fin_2d",
        "python heat_sink_inverse.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/three_fin_2d",
        "python heat_sink.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/three_fin_3d",
        "python three_fin_flow.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/three_fin_3d",
        "python three_fin_thermal.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/turbulent_channel/2d",
        "python re590_k_ep_LS.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/turbulent_channel/2d",
        "python re590_k_om_LS.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/turbulent_channel/2d_std_wf",
        "python re590_k_ep.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/turbulent_channel/2d_std_wf",
        "python re590_k_om.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/turbulent_channel/2d_std_wf",
        "python u_tau_lookup.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/wave_equation",
        "python wave_1d_causal.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/wave_equation",
        "python wave_1d.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/wave_equation",
        "python wave_inverse.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/waveguide/cavity_2D",
        "python waveguide2D_TMz.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/waveguide/cavity_3D",
        "python waveguide3D.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/waveguide/slab_2D",
        "python slab_2D.py training.max_steps=600",
        "popd",
    ],
    [
        "pushd examples/waveguide/slab_3D",
        "python slab_3D.py training.max_steps=600",
        "popd",
    ],
]


RUN_CMD_TEMPLATES = """#!/usr/bin/env bash

# Test training benchmark for a model.
function _set_params(){{
    model_item=${{1:-"model_item"}}   # (必选) 模型 item |fastscnn|segformer_b0| ocrnet_hrnetw48
    base_batch_size=${{2:-"1"}}       # (必选) 如果是静态图单进程，则表示每张卡上的BS，需在训练时*卡数
    fp_item=${{3:-"fp32"}}            # (必选) fp32|fp16
    run_mode=${{4:-"DP"}}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${{5:-"N1C1"}}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）

    backend="pytorch"
    model_repo="modulus"          # (必选) 模型套件的名字
    speed_unit="ms"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="time/iteration:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"

#   以下为通用执行命令，无特殊可不用修改
    model_name=${{model_item}}_bs${{base_batch_size}}_${{fp_item}}_${{run_mode}}  # (必填) 且格式不要改动,与竞品名称对齐
    device=${{CUDA_VISIBLE_DEVICES//,/ }}
    arr=(${{device}})
    num_gpu_devices=${{#arr[*]}}
    run_log_path=${{TRAIN_LOG_DIR:-$(pwd)}}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    speed_log_path=${{LOG_PATH_INDEX_DIR:-$(pwd)}}
    # deepxde_Eular_beam_bs1_fp32_DP_N1C1_log
    train_log_file=${{run_log_path}}/${{model_repo}}_${{model_name}}_${{device_num}}_log
    speed_log_file=${{speed_log_path}}/${{model_repo}}_${{model_name}}_${{device_num}}_speed
}}

function _analysis_log(){{
    echo "train_log_file: ${{train_log_file}}"
    echo "speed_log_file: ${{speed_log_file}}"
    cmd="python analysis_log.py --filename ${{train_log_file}} \\
        --speed_log_file ${{speed_log_file}} \\
        --model_name ${{model_name}} \\
        --base_batch_size ${{base_batch_size}} \\
        --run_mode ${{run_mode}} \\
        --fp_item ${{fp_item}} \\
        --keyword ${{keyword}} \\
        --skip_steps ${{skip_steps}} \\
        --device_num ${{device_num}} "
    echo ${{cmd}}
    eval $cmd
}}

function _train(){{
    echo "current CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES}}, model_name=${{model_name}}, device_num=${{device_num}}, is profiling=${{profiling}}"

#   以下为通用执行命令，无特殊可不用修改

    export DDE_BACKEND=pytorch
    train_cmd="{train_cmd}"
    echo "pwd: $PWD train_cmd: ${{train_cmd}} log_file: ${{train_log_file}}"
    set -x
    timeout 15m bash -c "${{train_cmd}}" > ${{train_log_file}} 2>&1
    if [ $? -ne 0 ];then
        echo -e "Generate ${{model_name}}, FAIL"
    else
        echo -e "Generate ${{model_name}}, SUCCESS"
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{{print $2}}'`
}}

_set_params $@
export frame_version=`python -c "import torch;print(torch.__version__)"`
echo "---------frame_version is ${{frame_version}}"
echo "---------Model commit is ${{model_commit}}"
echo "---------model_branch is ${{model_branch}}"

job_bt=`date '+%Y%m%d%H%M%S'`
_train
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${{job_et}}-${{job_bt}}))
_analysis_log
"""


def gen_end_to_end_shells():
    assert len(shell_content) == 51
    sset = set()
    for i, cmd_list in enumerate(shell_content):
        train_cmd = "; ".join(cmd_list)
        pushd_cmd, execute_cmd, popd_cmd = cmd_list

        example_name = pushd_cmd.split(" ")[-1]
        example_name = "/".join(example_name.split("/")[1:])
        example_name = example_name.replace("/", "-")
        file_name = execute_cmd.split(" ")[1].split(".")[0]
        example_name = "-".join([example_name, file_name])
        assert example_name not in sset, f"{example_name} already exists!"
        sset.add(example_name)

        os.makedirs(osp.join(example_name, "N1C1"), exist_ok=True)

        # generate files in benchmark_common
        os.makedirs(example_name, exist_ok=True)
        os.makedirs(osp.join(example_name, "benchmark_common"), exist_ok=True)
        ## generate benchmark_common/prepare.sh
        with open(osp.join(example_name, "benchmark_common", "prepare.sh"), "w") as f:
            f.write("# install pytorch\n")
            f.write("pip install torch --index-url https://download.pytorch.org/whl/cu118\n\n")
            # install modulus
            f.write("# install modulus\n")
            f.write("pushd ../modulus/\n")
            f.write("pip install -e .\n")
            f.write("popd\n\n")
            f.write("# install modulus-sym\n")
            # install modulus-sym
            f.write("pip install -e .\n")

        ## generate benchmark_common/analysis_log.py
        shutil.copy(
            "/workspace/hesensen/PaddleScience_enn_debug/benchmark/frame_benchmark/pytorch/dynamic/modulus-sym/scripts/analysis_log.py",
            osp.join(example_name, "benchmark_common", "analysis_log.py"),
        )

        ## generate benchmark_common/run_benchmark.sh
        with open(osp.join(example_name, "benchmark_common", "run_benchmark.sh"), "w") as f:
            # print(osp.join(example_name, "benchmark_common", "run_benchmark.sh"), train_cmd)
            f.write(RUN_CMD_TEMPLATES.format(train_cmd=train_cmd))

        # generate files in N1C1
        os.makedirs(osp.join(example_name, "N1C1"), exist_ok=True)
        ## generate N1C1/{example_name}.sh
        with open(osp.join(example_name, "N1C1", f"{example_name}_bs1_fp32_DP.sh"), "w") as f:
            # install modulus-sym
            f.write(f"model_item={example_name}\n")
            f.write("bs_item=1\n")
            f.write("fp_item=fp32\n")
            f.write("run_mode=DP\n")
            f.write("device_num=N1C1\n")
            f.write("pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple\n")
            f.write("# prepare\n")
            f.write("bash prepare.sh\n")
            f.write("# run\n")
            f.write("bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} 2>&1;\n")
            f.write("sleep 10;\n")


if __name__ == "__main__":
    gen_end_to_end_shells()
