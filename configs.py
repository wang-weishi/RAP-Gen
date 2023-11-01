import random
import torch
import logging
import multiprocessing
import numpy as np

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("--task", type=str, required=True,
                        choices=['tfix_rmd', 'tfix_rmd_rapgen', 'refine', 'refine_rapgen', 'refine_small',
                                 'refine_medium', 'selfapr',
                                 'selfapr_rapgen',
                                 'selfapr_rapgen_P1', 'selfapr_rapgen_P2', 'selfapr_rapgen_P3', 'selfapr_rapgen_P4',
                                 'selfapr_rapgen_P5',
                                 'selfapr_rapgen_P6', 'selfapr_rapgen_P7', 'selfapr_rapgen_P8', 'selfapr_rapgen_P9',
                                 'selfapr_rapgen_P11',
                                 'selfapr_rapgen_P12', 'selfapr_rapgen_P13', 'selfapr_rapgen_P14', 'selfapr_rapgen_P15',
                                 'selfapr_rapgen_P16'])
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--lang", type=str, default='')
    parser.add_argument("--eval_task", type=str, default='')
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 't5', 'bart', 'codet5'])
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--summary_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--res_fn", type=str, default='')
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--save_last_checkpoints", action='store_true')
    parser.add_argument("--always_save_model", action='store_true')
    parser.add_argument("--do_eval_bleu", action='store_true', help="Whether to evaluate bleu on dev set.")

    ## Required parameters
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--load_model_dir", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--tag_suffix", default='finetune', type=str,
                        help="Experiment full model tag suffix")
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--save_steps", default=-1, type=int, )
    parser.add_argument("--log_steps", default=-1, type=int, )
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    args = parser.parse_args()

    if args.task in ['refine_small', 'refine_medium', 'refine_rapgen','selfapr',
                     'selfapr_rapgen',
                     'selfapr_rapgen_P1', 'selfapr_rapgen_P2', 'selfapr_rapgen_P3', 'selfapr_rapgen_P4',
                     'selfapr_rapgen_P5',
                     'selfapr_rapgen_P6', 'selfapr_rapgen_P7', 'selfapr_rapgen_P8', 'selfapr_rapgen_P9',
                     'selfapr_rapgen_P11',
                     'selfapr_rapgen_P12', 'selfapr_rapgen_P13', 'selfapr_rapgen_P14', 'selfapr_rapgen_P15',
                     'selfapr_rapgen_P16']:
        args.lang = 'java'
    elif 'tfix' in args.task:
        args.lang = 'java-cs'
    return args


def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    cpu_cont = multiprocessing.cpu_count()
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    args.cpu_cont = cpu_cont


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True