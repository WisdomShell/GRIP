export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=29500 /path/to/your/GRIP/GRIP/inference/agent.py \
    --model_path /path/to/your/RL_model/global_step_xxx \
    --input_file /path/to/your/test_data/web.jsonl \
    --output_file /path/to/your/output/web.jsonl \
    --max_round 4 \
    --batch_size 32