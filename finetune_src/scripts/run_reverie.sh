name=DSRG_reverie
DATA_ROOT=../datasets

train_alg=dagger

features=vitbase
ft_dim=768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

outdir=${DATA_ROOT}/REVERIE/
pretrain_file=${DATA_ROOT}/REVERIE/pretrain/model_step_best_148000.pt

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert
      --train navigator
      --name ${name}

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200
      --max_objects 20

      --batch_size 8
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2

      --feat_dropout 0.5
      --dropout 0.5

      --model reverie
      
      --gamma 0.
      "

CUDA_VISIBLE_DEVICES='0' python -u reverie/main_nav_obj.py $flag  \
      --tokenizer bert \
      --bert_ckpt_file ${pretrain_file}
      # --eval_first