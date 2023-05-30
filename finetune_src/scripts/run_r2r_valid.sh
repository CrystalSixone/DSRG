name=DSRG_r2r_valid
DATA_ROOT=../datasets

train_alg=dagger

features=vitbase
ft_dim=768

ngpus=1
seed=0

outdir=${DATA_ROOT}/R2R/
resume_file=${DATA_ROOT}/R2R/navigator/DSRG_r2r/ckpts/best_val_unseen

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert   
      --name ${name}   
      --train valid
      --model r2r

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 4
      --lr 5e-6
      --iters 100000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.5
      --dropout 0.5
      
      --gamma 0.
      --resume_file ${resume_file}
      "

CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
      --tokenizer bert \
      --submit
