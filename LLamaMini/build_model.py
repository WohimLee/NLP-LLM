from datasets import load_dataset
from typing import Dict, List
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoConfig
import torch
import os
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling,Seq2SeqTrainingArguments,HfArgumentParser
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def print_model_parameters(model):
    # 打印模型参数的函数，输入是一个模型对象

    print('Layer Name & Parameters')    # 打印标题：层名称和参数
    print('----------------------------')

    total_params = 0    # 初始化变量，用于累计模型的总参数量

    for name, parameter in model.named_parameters():
        # 遍历模型的每个参数（通过 named_parameters() 获取名称和参数值）
        
        param_size = parameter.size()   # 获取当前参数的尺寸，即张量的形状
        
        # 计算当前参数的总数量，使用 torch.prod() 计算尺寸各维度的乘积
        param_count = torch.prod(torch.tensor(param_size)).item()   
        
        # 将当前参数的数量累加到总参数量中
        total_params += param_count
        
        # 打印当前层的名称、参数尺寸和参数数量，格式化输出便于对齐显示
        print(f'{name:50} | Size: {str(param_size):30} | Count: {str(param_count):20}')
        

    print('----------------------------')
    # 打印模型的总参数数量，并以百万(M)为单位格式化输出
    print(f'Total Parameters: {total_params} ({total_params / 1000000:.1f} M)')


def inference(  
    model: AutoModelForCausalLM,  # 定义推理函数，输入为一个语言模型
    tokenizer: AutoTokenizer,  # 使用的分词器
    input_text: str = 'once ',  # 输入文本的默认值为 'once '
    max_new_tokens: int = 16  # 最多生成的新 token 数量，默认为 16
):
    inputs = tokenizer(input_text, return_tensors='pt').to(device)  
    # 将输入文本 token 化，并将其转换为 PyTorch 张量，同时转移到指定设备（如 GPU 或 CPU）

    outputs = model.generate(
        **inputs,  # 将 token 化的输入传入模型
        pad_token_id=tokenizer.eos_token_id,  # 填充 token 的 ID 设置为 eos_token_id
        max_new_tokens=max_new_tokens,  # 设置生成的新 token 的最大数量
        do_sample=True,  # 启用采样而非贪心搜索
        top_k=40,  # top-k 采样：每次只从概率最高的 40 个 token 中选择
        top_p=0.95,  # top-p 采样：选择概率累计到 0.95 的 token
        temperature=0.8  # 温度设置为 0.8，控制生成的多样性（值越低越保守）
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  
    # 将生成的 token 序列解码为文本，并跳过特殊 token

    # print(outputs)  # 打印生成的 token 序列（可选调试行）
    print(generated_text)  # 打印生成的文本


# 定义处理函数，输入是一个包含示例的字典，键为字符串，值为列表
def process_func(examples: Dict[str, List]) :

    # 设置最长 token 数为 2048，对于当前任务，这个值足够大，通常不会超出限制
    max_token = 2048    

    # 使用 tokenizer 对示例中的文本进行编码，且不添加特殊 token（如 <CLS>, <SEP> 等）
    encoded_texts = tokenizer(examples['text'], add_special_tokens=False)
    
    # 提取出编码后的 token 序列列表，`input_ids_list` 是一个嵌套列表，其中每个元素是文本的 token 序列
    input_ids_list = encoded_texts['input_ids']

    # 初始化两个空列表，用于存储新的 input_ids 和 attention mask
    new_input_ids_list, new_attn_mask_list = [], []

    # 遍历每个文本的 token 序列
    for input_ids in input_ids_list:

        # 截取最后的 max_token-1 个 token，并在末尾添加 eos_token_id (表示序列结束)
        temp = input_ids[-max_token+1:] + [tokenizer.eos_token_id]
        
        # 将处理后的 token 序列添加到新的 input_ids 列表中
        new_input_ids_list.append(temp)
        
        # 创建与 temp 相同长度的 attention mask，1 表示这些位置的 token 有效
        new_attn_mask_list.append([1] * len(temp))

    # 返回新的 input_ids 列表和对应的 attention mask 列表
    return {
        'input_ids': new_input_ids_list,
        'attention_mask': new_attn_mask_list
    }

# 从指定路径加载预训练的 tokenizer，`trust_remote_code=True` 表示信任远程代码库
tokenizer = AutoTokenizer.from_pretrained('/home/ma-user/work/01440561/llama_test/qwen_tokenizer', trust_remote_code=True)


if __name__ == "__main__":

    hidden_size = 256

    # 中间层取 8/3 倍，按 128 向上取整
    intermediate_size = (int(hidden_size * 8/3 / 128) + 1) * 128

    # 只改动我们需要调整的参数，其余保持不变
    config = AutoConfig.for_model(
        model_type='llama',         # 模型类型为 'llama'
        vocab_size=len(tokenizer),  # 词汇表大小，等于 tokenizer 中的词汇数量
        hidden_size=hidden_size,    # 设置隐藏层大小
        intermediate_size=intermediate_size,  # 设置中间层大小
        num_attention_heads=16,     # 设置注意力头的数量为 16
        num_hidden_layers=4,        # 设置隐藏层数量为 4
        num_key_value_heads=8       # 设置键值头的数量为 8
    )
    print(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 使用全精度训练（torch.float32）
    model = AutoModelForCausalLM.from_config(config,torch_dtype=torch.float32).to(device)

    print(model)

    # 打印模型的每一层及其参数大小
    print_model_parameters(model)
    
    # 使用给定的输入文本进行推理
    inference(model, tokenizer,"我们")

    # 加载数据集
    # dataset_name_or_path = '1_poem_0.05k.jsonl'  # 可以替换为本地文件夹路径

    # 加载 JSON 格式的数据集
    all_data = load_dataset('json', data_files='/home/ma-user/work/01440561/llama_test/pretrain_vtest_zsd3.json')
    # 将数据集按 9:1 比例进行拆分，90% 为训练集，10% 为验证集
    all_data = all_data["train"].train_test_split(test_size=0.1)

    ds_train = all_data['train']
    ds_val = all_data['test']

    print(ds_train[:2])
    print(ds_val)

    num_proc = 4  # 处理数据时所用的线程数

    # 对训练集应用 process_func 函数，进行批处理并多线程加速
    ds_train = ds_train.map(
        process_func,       # 应用的处理函数
        batched=True,       # 批量处理数据
        num_proc=num_proc,  # 使用多线程处理
        remove_columns=ds_train.column_names,   # 移除原始列
        desc='Running tokenizer on train_set: ' # 显示进度描述
    )

    # 对验证集应用 process_func 函数，进行批处理并多线程加速
    ds_val = ds_val.map(
        process_func,       # 应用的处理函数
        batched=True,       # 批量处理数据
        num_proc=num_proc,  # 使用多线程处理
        remove_columns=ds_val.column_names,     # 移除原始列
        desc='Running tokenizer on val_set: '   # 显示进度描述
    )

    print(ds_train)
    print(ds_val)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='saves',         # 输出路径，包括模型检查点、中间文件等
        overwrite_output_dir=True,  # 是否覆写 output_dir
        do_train=True,  # 是否做训练
        do_eval=True,   # 是否做评估
        eval_steps=100, # 评估步骤间隔
        eval_strategy="steps",
        per_device_train_batch_size=4,  # 每设备批次
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,  # 梯度累计步大小，省显存，但小模型没必要，用 1 收敛比较快
        learning_rate=1e-1,             # 学习率大小
        lr_scheduler_type='cosine',     # 学习率调度策略，LLM 训练一般都用余弦
        bf16=torch.cuda.is_bf16_supported(),        # 尝试配置 bf16
        fp16=not torch.cuda.is_bf16_supported(),    # bf16 不行就上 fp16
        logging_steps=1,        # 打印步骤间隔
        report_to=None,         # 日志输出目标，不想用 wandb 可以设置为 None
        num_train_epochs=30,    # 训练轮数，2 ~ 3 即可
        save_steps=1000,        # 检查点保存步骤间隔
        save_total_limit=2,     # output_dir 内留存的检查点最大数目
        save_only_model=True,
        seed=3407               # 随机种子
    )

    trainer = Trainer(
        model=model,                # 模型实例
        args=training_args,         # 训练参数
        train_dataset=ds_train,     # 训练集
        eval_dataset=ds_val,  # 验证集（评估集）
        tokenizer=tokenizer,  # 分词器
        data_collator=data_collator,  # data collator
    )
    trainer.train()

    inference(model,tokenizer,'我们',max_new_tokens=256)
