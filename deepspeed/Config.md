


```json
{
  "train_batch_size": "auto", // 定义训练过程中一次迭代的总批次大小（包括所有GPU）。设为 "auto" 表示 DeepSpeed 将自动计算和调整该值。
  "train_micro_batch_size_per_gpu": "auto", // 定义每个 GPU 上的一次微批处理大小。微批处理是指在单次反向传播中处理的样本数量。设为 "auto" 表示 DeepSpeed 自动确定合适的值
  "gradient_accumulation_steps": "auto",    // 定义在一次反向传播更新前累积的梯度步数。通过增加这个值，可以使用更大的有效批次大小，而不需要增加显存需求。设为 "auto" 表示 DeepSpeed 自动设置这个值
  "gradient_clipping": "auto",              // 定义梯度裁剪的阈值，用于防止梯度爆炸。设为 "auto" 表示 DeepSpeed 自动确定是否启用和设置裁剪阈值
  "zero_allow_untested_optimizer": true,    // 如果设置为 true，即使使用未经过 DeepSpeed 测试的优化器，Zero Redundancy Optimizer (ZeRO) 也将允许使用。ZeRO 是 DeepSpeed 中用于高效处理大规模模型的关键优化技术
  "fp16": { // FP16（半精度浮点数）相关设置
    "enabled": "auto",
    "loss_scale": 0,            // 定义损失缩放因子，0 表示使用动态损失缩放，以避免 FP16 计算中的数值下溢
    "loss_scale_window": 1000,  // 损失缩放因子的窗口大小，即在更新损失缩放之前，累积多少个步骤的统计信息
    "initial_scale_power": 16,  // 初始的损失缩放值为 2^16
    "hysteresis": 2,            // 在减少损失缩放因子之前的容忍度（hysteresis），避免过早减少损失缩放
    "min_loss_scale": 1         // 最小的损失缩放因子，防止缩放因子降得过低
  },
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {    // ZeRO 优化相关设置
    "stage": 3,             // ZeRO 优化的阶段，Stage 3 是最先进的阶段，涉及最大程度的内存优化
    "offload_optimizer": {  
      "device": "cpu",      // 将优化器状态（例如动量和二阶矩）卸载到 CPU 以节省 GPU 内存
      "pin_memory": true    // 启用固定内存以加快 CPU 和 GPU 之间的数据传输
    },
    "offload_param": {
      "device": "cpu",      // 将模型参数卸载到 CPU 以节省 GPU 内存
      "pin_memory": true    // 启用固定内存以加快数据传输
    },
    "overlap_comm": true,         // 允许通信和计算重叠，以提高效率
    "contiguous_gradients": true, // 将梯度存储为连续内存块，以减少内存碎片并提高性能
    "sub_group_size": 1e9,        // 定义参数子组的大小，以优化参数更新过程中的通信开销
    "reduce_bucket_size": "auto", // 定义梯度归约的桶大小。设为 "auto" 表示 DeepSpeed 自动调整这个大小以优化性能
    "stage3_prefetch_bucket_size": "auto",        // Stage 3 优化阶段中，参数预取的桶大小。设为 "auto" 表示自动调整
    "stage3_param_persistence_threshold": "auto", // 控制哪些参数在内存中保持持久存储。设为 "auto" 表示自动调整
    "stage3_max_live_parameters": 1e9,            // Stage 3 优化阶段中，允许在内存中存活的最大参数数量
    "stage3_max_reuse_distance": 1e9,             // 参数在被重新使用前的最大距离，用于优化内存重用
    "stage3_gather_16bit_weights_on_model_save": true // 在保存模型时，将16位权重收集在一起，以提高模型保存的效率和一致性
  }
}
```