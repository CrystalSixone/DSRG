import os,sys
root_path = os.getcwd()
current_path = os.path.join(root_path,'finetune_src')
sys.path.append(root_path)
sys.path.append(current_path)

import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import torch
import string
from tensorboardX import SummaryWriter
from transformers import DistilBertTokenizer

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from utils.data import ImageFeaturesDB, PickSpecificWords

from reverie.agent_obj import GMapObjectNavAgent
from reverie.data_utils import ObjectFeatureDB, construct_instrs, load_obj2vps, ObjectFeaatureDBv2
from reverie.env import ReverieObjectNavBatch
from reverie.parser import parse_args

def build_dataset(args, rank=0):
    tok = bert_tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    obj_db = ObjectFeatureDB(args.obj_ft_file, args.obj_feat_size) # REVERIE original
    obj_v2_db = ObjectFeaatureDBv2(args, args.obj_v2_ft_file,cat_file=args.cat_file) # More fine-grained object info
    obj2vps = load_obj2vps(os.path.join(args.anno_dir, 'BBoxes.json'))

    word_picker = PickSpecificWords(args.cat_file)

    if args.env_aug == 'env_edit':
        print('use env_edit features!!')
        envedit_feat_db = ImageFeaturesDB(args.envedit_ft_file, args.image_feat_size)
        train_feat_db = [feat_db,envedit_feat_db]
    else:
        train_feat_db = feat_db

    dataset_class = ReverieObjectNavBatch

    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            word_picker=word_picker
        )
        aug_env = dataset_class(
            train_feat_db, obj_db, aug_instr_data, args.connectivity_dir, obj2vps, 
            batch_size=args.batch_size, max_objects=args.max_objects,
            angle_feat_size=args.angle_feat_size, 
            seed=args.seed+rank, sel_data_idxs=None, name='aug', 
            multi_endpoints=args.multi_endpoints, multi_startpoints=args.multi_startpoints,tok=tok,args=args,
            obj_v2_db=obj_v2_db
        )
    else:
        aug_env = None
        aug_instr_data = None

    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], 
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
        word_picker=word_picker
    )
    train_env = dataset_class(
        train_feat_db, obj_db, train_instr_data, args.connectivity_dir, obj2vps,
        batch_size=args.batch_size, max_objects=args.max_objects,
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train', 
        multi_endpoints=args.multi_endpoints, multi_startpoints=args.multi_startpoints,tok=tok,args=args,
        obj_v2_db=obj_v2_db
    )

    val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    if args.submit:
        val_env_names.append('test')
        
    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            word_picker=word_picker
        )
        val_env = dataset_class(
            feat_db, obj_db, val_instr_data, args.connectivity_dir, obj2vps, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
            max_objects=None, multi_endpoints=False, multi_startpoints=False,tok=tok,args=args,
            obj_v2_db=obj_v2_db
        )   # evaluation using all objects
        val_envs[split] = val_env

    return train_env, val_envs, aug_env, aug_instr_data, bert_tok

def train(args, train_env, val_envs, rank=-1, tok=None):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = GMapObjectNavAgent
    listner = agent_class(args, train_env, rank=rank,tok=tok)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration ".format(args.resume_file, start_iter),
                record_file
            )
       
    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            prefix = 'submit' if args.detailed_output is False else 'detail'
            output_file = os.path.join(args.pred_dir, "%s_%s_%s.json" % (
                prefix, env_name, args.fusion))
            if os.path.exists(output_file):
                continue
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results(detailed_output=args.detailed_output)
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            
            if default_gpu and env_name != 'test':
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
            
            for i,_ in enumerate(preds):
                preds[i]["predObjId"] = preds[i].pop("pred_objid")
                if preds[i]["predObjId"] is not None:
                    preds[i]["predObjId"] = int(preds[i]["predObjId"])
            
            if args.submit:
                with open(output_file,'w') as f:
                    json.dump(preds, f, sort_keys=True, indent=4, separators=(',', ': '))
                print('submit file has been saved in {}.'.format(output_file))

        if default_gpu:
            write_to_record_file(loss_str, record_file)

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    # best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":""}}
    best_val = {'val_unseen': {"rgs": 0., "rgspl": 0., "state":""}}

    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        # Train for log_every interval
        listner.env = train_env
        listner.train(interval, feedback=args.feedback)  # Train interval iters
        
        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            OG_loss = sum(listner.logs['OG_loss']) / max(len(listner.logs['OG_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/OG_loss", OG_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, OG_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, OG_loss, policy_loss, critic_loss),
                record_file
            )

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, env in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

                # select model by rgs + rgspl
                if env_name in best_val:
                    if score_summary['rgs'] + score_summary['rgspl'] >= best_val[env_name]['rgs'] + score_summary['rgspl']:
                        best_val[env_name]['rgs'] = score_summary['rgs']
                        best_val[env_name]['rgspl'] = score_summary['rgspl']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        listner.save(idx, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))
                
        
        if default_gpu:
            listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))

            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)


def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = GMapObjectNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        prefix = 'submit' if args.detailed_output is False else 'detail'
        output_file = os.path.join(args.pred_dir, "%s_%s_%s.json" % (
            prefix, env_name, args.fusion))
        if os.path.exists(output_file):
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n', record_file)

            if args.submit:
                json.dump(
                    preds, open(output_file, 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
                print('pred file has been saved in {}.'.format(output_file))

def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env, aug_data, bert_tok = build_dataset(args, rank=rank)
            
    if args.train == 'navigator':
        train(args, train_env, val_envs, rank=rank,tok=bert_tok)
    elif args.train == 'valid':
        valid(args, train_env, val_envs, rank=rank)

if __name__ == '__main__':
    main()