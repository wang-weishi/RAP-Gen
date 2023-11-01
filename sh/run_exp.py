#!/usr/bin/env python
import os
import argparse

def get_cmd(task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch, warmup,
            model_dir, summary_dir, res_fn, load_model_dir, tag_suffix):
    cmd_str = 'bash exp_with_args.sh %s %s %s %d %d %d %d %d %d %d %d %d %s %s %s %s %s' % \
              (task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch,
               warmup, model_dir, summary_dir, res_fn, load_model_dir, tag_suffix)
    return cmd_str


def get_args_by_task_model(task, sub_task, model_tag):
    if task in ['tfix_rmd', 'tfix_rmd_rapgen']:
        src_len = 400
        trg_len = 256
        epoch = 40
        patience = 40
    elif task == 'refine':
        # small: Read 46680 examples, avg src len: 31, avg trg len: 28, max src len: 50, max trg len: 50
        # [TOKENIZE] avg src len: 50, avg trg len: 45, max src len: 129, max trg len: 121
        # medium:  Read 52364 examples, avg src len: 74, avg trg len: 73, max src len: 100, max trg len: 100
        # [TOKENIZE] avg src len: 117, avg trg len: 114, max src len: 238, max trg len: 238
        if sub_task == 'small':
            src_len = 130
            trg_len = 120
        elif sub_task == 'medium':
            src_len = 240
            trg_len = 240
        epoch = 50
        patience = 5
    elif task == 'refine_small':
        src_len = 130
        trg_len = 120
        epoch = 30
        patience = 30
    elif task == 'refine_medium':
        src_len = 240
        trg_len = 240
        epoch = 30
        patience = 30
    elif task == 'refine_rapgen':
        # mdedium [TOKENIZE] avg src len: 380, avg trg len: 114, max src len: 622, max trg len: 238
        # small [TOKENIZE] avg src len: 167, avg trg len: 45, max src len: 367, max trg len: 121
        if sub_task == 'small':
            src_len = 260
            trg_len = 120
        elif sub_task == 'medium':
            src_len = 400
            trg_len = 240
        epoch = 30
        patience = 30
    elif task == 'selfapr':
        src_len = 400
        trg_len = 100
        epoch = 10
        patience = 10
    elif 'selfapr_rapgen' in task:
        src_len = 512
        trg_len = 100
        epoch = 10
        patience = 10
    if 'codet5_small' in model_tag:
        bs = 32
        if (task == 'refine' and sub_task == 'small') or 'tfix' in task or (
                task == 'refine_rapgen' and sub_task == 'small'):
            bs = 64
        elif task == 'refine_small':
            bs = 64
    elif 'codet5_large' in model_tag:
        bs = 8
    else:
        bs = 32
        if (task == 'refine' and sub_task == 'medium') or (task == 'refine_rapgen' and sub_task == 'medium'):
            bs = 25
        elif task == 'refine_medium':
            bs = 25
        elif 'tfix' in task:
            bs = 25
        elif 'selfapr' in task:
            bs = 25

    # lr = 5
    if 'codet5_base' in model_tag and 'refine_small' in task:
        lr = 5
    elif 'codet5_base' in model_tag and 'refine' in task and sub_task == 'small':
        lr = 5
    elif 'tfix' in task:
        lr = 10
    elif 'refine' in task:
        lr = 10
    else:
        lr = 5

    return bs, lr, src_len, trg_len, patience, epoch


def run_one_exp(args):
    bs, lr, src_len, trg_len, patience, epoch = get_args_by_task_model(args.task, args.sub_task, args.model_tag)
    print('============================Start Running==========================')
    cmd_str = get_cmd(task=args.task, sub_task=args.sub_task, model_tag=args.model_tag, gpu=args.gpu,
                      data_num=args.data_num, bs=bs, lr=lr, source_length=src_len, target_length=trg_len,
                      patience=patience, epoch=epoch, warmup=1000,
                      model_dir=args.model_dir, summary_dir=args.summary_dir,
                      res_fn='{}/{}_{}.txt'.format(args.res_dir, args.task, args.model_tag),
                      load_model_dir=args.load_model_dir, tag_suffix=args.tag_suffix)
    print('%s\n' % cmd_str)
    os.system(cmd_str)


def get_sub_tasks(task):
    if task in ['refine', 'refine_rapgen']:
        sub_tasks = ['small', 'medium']
    elif task in ['tfix_rmd', 'tfix_rmd_rapgen', 'refine_small', 'refine_medium', 'selfapr', 'selfapr_rapgen',
                  'selfapr_rapgen_P1', 'selfapr_rapgen_P2', 'selfapr_rapgen_P3', 'selfapr_rapgen_P4',
                  'selfapr_rapgen_P5',
                  'selfapr_rapgen_P6', 'selfapr_rapgen_P7', 'selfapr_rapgen_P8', 'selfapr_rapgen_P9',
                  'selfapr_rapgen_P11',
                  'selfapr_rapgen_P12', 'selfapr_rapgen_P13', 'selfapr_rapgen_P14', 'selfapr_rapgen_P15',
                  'selfapr_rapgen_P16']:
        sub_tasks = ['none']
    return sub_tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, default='codet5_small',
                        choices=['roberta', 'codebert', 'bart_base', 't5_base', 'codet5_small', 'codet5_base',
                                 'codet5_large'])
    parser.add_argument("--task", type=str, default='tfix_rmd',
                        choices=['tfix_rmd', 'tfix_rmd_rapgen', 'refine', 'refine_rapgen', 'refine_small',
                                 'refine_medium', 'selfapr',
                                 'selfapr_rapgen',
                                 'selfapr_rapgen_P1', 'selfapr_rapgen_P2', 'selfapr_rapgen_P3', 'selfapr_rapgen_P4',
                                 'selfapr_rapgen_P5',
                                 'selfapr_rapgen_P6', 'selfapr_rapgen_P7', 'selfapr_rapgen_P8', 'selfapr_rapgen_P9',
                                 'selfapr_rapgen_P11',
                                 'selfapr_rapgen_P12', 'selfapr_rapgen_P13', 'selfapr_rapgen_P14', 'selfapr_rapgen_P15',
                                 'selfapr_rapgen_P16'])
    parser.add_argument("--sub_task", type=str, default='none')
    parser.add_argument("--res_dir", type=str, default='results', help='directory to save fine-tuning results')
    parser.add_argument("--model_dir", type=str, default='saved_models', help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='tensorboard', help='directory to save tensorboard summary')
    parser.add_argument("--data_num", type=int, default=-1, help='number of data instances to use, -1 for full data')
    parser.add_argument("--gpu", type=int, default=0, help='index of the gpu to use in a cluster')
    parser.add_argument("--load_model_dir", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--tag_suffix", default='finetune', type=str,
                        help="Experiment full model tag suffix")

    args = parser.parse_args()

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    assert args.sub_task in get_sub_tasks(args.task)

    run_one_exp(args)
