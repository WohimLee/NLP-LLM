# LLama-Factory

- GitHub: https://github.com/hiyouga/LLaMA-Factory
>环境搭建
```sh
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" # 千万别乱加 -i 源
pip install deepspeed # 必装
pip install zhipuai nvitop # 选装
```


- 训练说明: https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/README.md

>全量微调
```sh
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
```


## 自己数据集
>数据集config
- 路径: LLaMA-Factory/data
/dataset_info.json
- 在这里添加自己数据集的配置信息
```json
{
  "identity": {
    "file_name": "identity.json"
  },
  "alpaca_en_demo": {
    "file_name": "alpaca_en_demo.json"
  },
  ...
}
```

```yaml

```

>训练config
- 路径: LLaMA-Factory/examples/train_full
/llama3_full_sft_ds3.yaml
```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct # 选择模型，如 ChatGLM, QWen

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: identity,alpaca_en_demo # 训练的数据集名称
template: llama3 # 看语言选, 中文用 qwen
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/full/sft # 模型输出目录, 会非常大
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1  # 每个卡的batch
gradient_accumulation_steps: 2  # 梯度累积
learning_rate: 1.0e-5           # 学习率
num_train_epochs: 3.0           # 轮数
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
# total batch = 卡数 * 每个卡的batch * 梯度累积

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```
