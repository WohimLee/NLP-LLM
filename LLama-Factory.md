# LLama-Factory

- GitHub: https://github.com/hiyouga/LLaMA-Factory
>环境搭建
- 清华源: -i https://pypi.tuna.tsinghua.edu.cn/simple 
- 阿里源: --index-url https://mirrors.aliyun.com/pypi/simple
```sh
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

conda create -n azen python=3.9

cd LLaMA-Factory
pip install -e ".[torch,metrics]" # 千万别乱加 -i 源
pip install deepspeed # 必装
pip install zhipuai nvitop # 选装
```


- 训练说明: https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/README.md




## 训练自己的数据集
### 1 更改配置信息
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
model_name_or_path: /home/ma-user/work/model/t # 选择模型，如 ChatGLM, QWen

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: identity,alpaca_en_demo # 训练的数据集名称
template: llama3 # 看语言选, 中文用 qwen
cutoff_len: 1024
max_samples: 1000 # 最大训练的条数, 数据集有2700, 想全部跑完就得写2700以上
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

### 2 启动训练
>全量微调
```sh
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
```

### 3 推理
>推理参数
- 路径: checkpoint-[迭代次数]/generation_config.json
```json
{
  "bos_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [
    151645,
    151643
  ],
  "pad_token_id": 151643,
  "repetition_penalty": 1.05,
  "temperature": 0.7,
  "top_k": 20,
  "top_p": 0.8,
  "transformers_version": "4.43.3"
}
```

去模型官网看推理脚本
- Model Scope: https://www.modelscope.cn/home
- Hugging Face: 

```py
# from modelscope import AutoModelForCausalLM, AutoTokenizer # 这里改成 transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto
model_name_or_path = "qwen/Qwen2-7B"
model = AutoModelForCausalLM.from_pretrained(
  model_name_or_path,
  torch_dtype="auto",
  device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

prefix = "北京是中国的首都"
model_inputs = tokenizer([prefix], return_tensors="pt").to(device)

generated_ids = model.generate(
  model_inputs.input_ids,
  max_new_tokens=400,
  repetition_penalty=1.15
)
generated_ids = [
  output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```
