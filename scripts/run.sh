
# llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
python -m torch.distributed.launch \
  --nproc_per_node 8 \
  --master_port 12332 \
  ./src/train.py \
    --deepspeed ./examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path /home/ma-user/work/model/t \
    --dataset_dir ./data \
    --dataset identity \
    --template qwen \
    --finetuning_type full \
    --output_dir /home/ma-user/work/01449344/output/out2  \
    --overwrite_cache true \
    --overwrite_output_dir true \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --warmup_ratio 0.1 \
    --learning_rate 2e-5 \
    --num_train_epochs 5.0 \
    --plot_loss \
    --bf16
    
    
# torchrun --nproc_per_node=8 \
#   ./src/train.py \
#     --deepspeed ./examples/deepspeed/ds_z3_config.json \
#     --stage sft \
#     --do_train \
#     --model_name_or_path /home/ma-user/modelarts/inputs/base_model_path_0 \
#     --dataset_dir ./data \
#     --dataset identity \
#     --template qwen \
#     --finetuning_type full \
#     --output_dir /home/ma-user/modelarts/outputs/save_model_path_0  \
#     --overwrite_cache true \
#     --overwrite_output_dir true \
#     --cutoff_len 4096 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --logging_steps 1 \
#     --warmup_ratio 0.1 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 5.0 \
#     --plot_loss \
#     --bf16