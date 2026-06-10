# H20 与昇腾 910B Profiler 使用说明

本文介绍如何在 NVIDIA H20 和昇腾 910B 上运行 nanochat 预训练 Profiler，采集训练过程中的算子耗时、CPU/设备活动和模块级标记。

本文涉及以下文件：

```text
nanochat/gpt_profiler.py
scripts/base_train_profiler.py
scripts/base_train_profiler_npu.py
runs/run_base_profiler.sh
runs/run_base_profiler_npu.sh
```

其中：

- `run_base_profiler.sh` 是 H20 启动脚本。
- `run_base_profiler_npu.sh` 是昇腾 910B 启动脚本。
- `base_train_profiler.py` 使用 `torch.profiler` 采集 CUDA Profiler 数据。
- `base_train_profiler_npu.py` 使用 `torch_npu.profiler` 采集 NPU Profiler 数据。
- `gpt_profiler.py` 在模型内部加入了 `record_function` 标记，用于观察 attention、MLP 和各 Transformer block 的耗时。

下文中的 `run.sh` 和 `run_npu.sh` 只是方便设置环境变量的本地启动模板，不需要上传到 GitHub。

## 1. H20 Profiler

### 1.1 激活环境

```bash
cd /path/to/nanochat
source nanochat-h20/bin/activate
```

### 1.2 H20 启动模板

可以在本地创建 `runs/run.sh`，内容如下：

```bash
RUN_MODE=compile \
GPUS=0 \
NPROC=1 \
DEPTH=24 \
MAX_SEQ_LEN=2048 \
DEVICE_BATCH_SIZE=1 \
TOTAL_BATCH_SIZE=2048 \
PROFILE_STEPS=5 \
PROFILE_WAIT=2 \
PROFILE_WARMUP=2 \
PROFILE_ACTIVE=1 \
PROFILE_REPEAT=1 \
PROFILE_MODE=sampled \
USE_SWANLAB=1 \
SWANLAB_PROJECT="nanochat-profiler" \
SWANLAB_MODE="local" \
SWANLAB_LOGDIR="./swanlog" \
SWANLAB_EXPERIMENT_NAME="d24_s2048_5steps_compile_profiler" \
SWANLAB_TAGS="nanochat,profiler,d24,compile,5steps" \
bash ./runs/run_base_profiler.sh

# 双卡示例：
# RUN_MODE=compile \
# GPUS=0,1 \
# NPROC=2 \
# DEPTH=24 \
# MAX_SEQ_LEN=2048 \
# DEVICE_BATCH_SIZE=1 \
# TOTAL_BATCH_SIZE=4096 \
# PROFILE_STEPS=5 \
# PROFILE_WAIT=2 \
# PROFILE_WARMUP=2 \
# PROFILE_ACTIVE=1 \
# PROFILE_REPEAT=1 \
# PROFILE_MODE=sampled \
# USE_SWANLAB=1 \
# SWANLAB_PROJECT="nanochat-profiler" \
# SWANLAB_MODE="local" \
# SWANLAB_LOGDIR="./swanlog" \
# SWANLAB_EXPERIMENT_NAME="d24_s2048_5steps_compile_2gpus_profiler" \
# SWANLAB_TAGS="nanochat,profiler,d24,compile,5steps,gpu2-3" \
# bash ./runs/run_base_profiler.sh
```

执行：

```bash
bash runs/run.sh
```

也可以不创建 `run.sh`，直接复制上面的环境变量和命令到终端执行。

### 1.3 查看 H20 输出

默认日志保存在：

```text
logs/
```

Profiler trace 默认保存在：

```text
profiler_traces/
└── base_d24_s2048_5steps_gpu1_ids6_sampled_compile_timing/
    └── rank0/
```

具体目录名会根据模型深度、序列长度、设备编号、运行模式和详细信息开关自动变化。

查看生成的文件：

```bash
find profiler_traces -maxdepth 5 -type f | sort
```

启动 TensorBoard：

```bash
tensorboard --logdir ./profiler_traces --host 0.0.0.0 --port 6006
```

如果任务运行在远程服务器上，可以在本地建立 SSH 端口转发：

```bash
ssh -L 6006:127.0.0.1:6006 user@server
```

然后在本地浏览器访问：

```text
http://127.0.0.1:6006
```

H20 的 `rank0` 目录中还会生成 `key_averages_step*.txt`，可以直接查看各算子的 CUDA 和 CPU 汇总耗时。

## 2. 昇腾 910B Profiler

### 2.1 激活环境

```bash
cd /path/to/nanochat
source nanochat-npu/bin/activate
```

如果尚未创建该环境，可以先使用 `uv` 创建：

```bash
uv venv nanochat-npu
source nanochat-npu/bin/activate
```

NPU 启动脚本会在文件存在时自动加载 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

脚本默认设置：

```bash
export NANOCHAT_DTYPE=bfloat16
```

### 2.2 910B 启动模板

可以在本地创建 `runs/run_npu.sh`，内容如下：

```bash
# 单卡示例：
# RUN_MODE=no_compile \
# NPUS=0 \
# NPROC=1 \
# DEPTH=24 \
# MAX_SEQ_LEN=2048 \
# DEVICE_BATCH_SIZE=1 \
# TOTAL_BATCH_SIZE=2048 \
# PROFILE_STEPS=5 \
# PROFILE_WAIT=2 \
# PROFILE_WARMUP=2 \
# PROFILE_ACTIVE=1 \
# PROFILE_REPEAT=1 \
# PROFILE_MODE=sampled \
# USE_SWANLAB=0 \
# SWANLAB_PROJECT="nanochat-profiler" \
# SWANLAB_MODE="local" \
# SWANLAB_LOGDIR="./swanlog" \
# SWANLAB_EXPERIMENT_NAME="d24_s2048_5steps_npu_no_compile_profiler" \
# SWANLAB_TAGS="nanochat,profiler,d24,npu,no_compile,5steps" \
# bash ./runs/run_base_profiler_npu.sh

# 双卡示例：
RUN_MODE=no_compile \
NPUS=0,1 \
NPROC=2 \
DEPTH=24 \
MAX_SEQ_LEN=2048 \
DEVICE_BATCH_SIZE=1 \
TOTAL_BATCH_SIZE=4096 \
PROFILE_STEPS=5 \
PROFILE_WAIT=2 \
PROFILE_WARMUP=2 \
PROFILE_ACTIVE=1 \
PROFILE_REPEAT=1 \
PROFILE_MODE=sampled \
USE_SWANLAB=1 \
SWANLAB_PROJECT="nanochat-profiler" \
SWANLAB_MODE="local" \
SWANLAB_LOGDIR="./swanlog" \
SWANLAB_EXPERIMENT_NAME="d24_s2048_5steps_npu2_ids0x1_no_compile_profiler" \
SWANLAB_TAGS="nanochat,profiler,d24,npu,no_compile,5steps,npu0-1" \
bash ./runs/run_base_profiler_npu.sh
```

执行：

```bash
bash runs/run_npu.sh
```

单卡时，`run_base_profiler_npu.sh` 使用普通 Python 进程：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python -m scripts.base_train_profiler_npu
```

多卡时，脚本自动使用 `torchrun` 和 HCCL：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  -m scripts.base_train_profiler_npu
```

### 2.3 查看 910B 输出

如果训练在远程服务器上，先将 `trace_view.json` 下载到本地。

在本地浏览器打开：

[https://www.ui.perfetto.dev/](https://www.ui.perfetto.dev/)

将下载到本地的 `trace_view.json` 文件拖入 Perfetto 页面，即可查看 910B Profiler 结果。默认只采集 rank 0，避免多卡 trace 占用过多磁盘空间。

## 3. 主要配置参数

### 3.1 设备和训练参数

| 参数 | H20 | 910B | 说明 |
| --- | --- | --- | --- |
| 设备编号 | `GPUS` | `NPUS` | 单卡填写一个编号，多卡使用逗号分隔 |
| 进程数 | `NPROC` | `NPROC` | 必须与参与训练的设备数量一致 |
| 模型深度 | `DEPTH` | `DEPTH` | Transformer 层数 |
| 序列长度 | `MAX_SEQ_LEN` | `MAX_SEQ_LEN` | 单条样本的 token 长度 |
| 单设备 batch | `DEVICE_BATCH_SIZE` | `DEVICE_BATCH_SIZE` | 每个设备每次 forward/backward 的样本数 |
| 全局 token batch | `TOTAL_BATCH_SIZE` | `TOTAL_BATCH_SIZE` | 每个 optimizer step 处理的 token 总数 |

启动前需要满足：

```text
TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NPROC) == 0
```

梯度累积步数为：

```text
GRAD_ACCUM_STEPS =
    TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NPROC)
```

例如双卡、`DEVICE_BATCH_SIZE=1`、`MAX_SEQ_LEN=2048`、`TOTAL_BATCH_SIZE=4096` 时：

```text
GRAD_ACCUM_STEPS = 4096 / (1 * 2048 * 2) = 1
```

### 3.2 Profiler 采样参数

| 参数 | 说明 |
| --- | --- |
| `PROFILE_STEPS` | 整个测试运行多少个 optimizer step 后停止 |
| `PROFILE_WAIT` | 开始采集前跳过的 step 数 |
| `PROFILE_WARMUP` | Profiler 预热 step 数，不写入最终 trace |
| `PROFILE_ACTIVE` | 每个采样周期实际记录的 step 数 |
| `PROFILE_REPEAT` | 采样周期重复次数 |
| `PROFILE_MODE` | `sampled` 或 `continuous` |

使用 `sampled` 时必须满足：

```text
PROFILE_STEPS >=
    PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE * PROFILE_REPEAT
```

本文示例为：

```text
5 = 2 + 2 + 1 * 1
```

对应过程：

1. 前 2 个 optimizer step 不采集。
2. 接下来 2 个 optimizer step 用于 Profiler 预热。
3. 最后 1 个 optimizer step 写入 trace。

推荐先使用 `sampled`。`continuous` 会记录整个运行过程，短任务可以使用，但模型较大或 step 较多时会生成很大的 trace 文件。
