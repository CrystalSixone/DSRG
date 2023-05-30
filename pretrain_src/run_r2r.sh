
name=DSRG_r2r_pretrain
DATA_ROOT=../datasets/R2R/
NODE_RANK=0
NUM_GPUS=1

# train
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    train_r2r.py --world_size ${NUM_GPUS} \
    --name ${name} \
    --vlnbert cmt \
    --model_config config/r2r_model_config.json \
    --config config/r2r_pretrain.json \
    --root_dir $DATA_ROOT
