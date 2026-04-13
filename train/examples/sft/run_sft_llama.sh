set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NAME=GRIPSFT_1
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=datasets/$NAME/train.parquet \
    data.val_files=datasets/$NAME/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-6 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=16 \
    model.partial_pretrain=/path/to/your/Meta-Llama-3-8B-Base \
    model.enable_gradient_checkpointing=false \
    trainer.default_local_dir=/path/to/your/SFT_model \
    trainer.project_name=GRIPSFT \
    trainer.experiment_name=$NAME \
    trainer.logger=['console'] \
    trainer.total_epochs=8 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true