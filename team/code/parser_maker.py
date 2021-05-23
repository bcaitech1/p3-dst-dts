from attrdict import AttrDict

""" wandb sweep를 위해 argparser 생성기
근데 나중에는 사용안해서 코드 잘되는지 모름
"""

conf_keys = AttrDict(
    train_batch_size=int,
    eval_batch_size=int,
    hidden_size=int,
    hidden_dim=int,
    max_label_length=int,
    vocab_size=int,
    random_seed=int,
    n_gate=int,
    attn_head=int,
    num_rnn_layers=int, 
    num_train_epochs=int,
    n_transformers=int,
    proj_dim=int,
    max_seq_length=int,
    learning_rate=float,
    warmup_ratio=float,
    weight_decay=float,
    max_grad_norm=float,
    adam_epsilon=float,
    hidden_dropout_prob=float,
    teacher_forcing_ratio=float,
    use_transformer=bool,
    use_amp=bool,
    use_small_data=bool,
    fix_utterance_encoder=bool,
    use_larger_slot_encoding=bool,
    zero_init_rnn=bool,
    ModelName=str,
    distance_metric=str,
    device_pref=str,
    data_dir=str,
    preprocessor=str,
    model_dir=str,
    model_name_or_path=str,
    task_name=str,
    model_class=str,
    
    use_domain_slot=str,
    use_decoder_ts=bool,
    decoder_n_heads=int,
    decoder_n_layers=int,
)

def update_parser(parser):
    for k, v in conf_keys.items():
        parser.add_argument(f'--{k}', type=v)

    return parser

def update_config(conf, args):
    for k, v in vars(args).items():
        if v is not None:
            setattr(conf, k, v)

    return conf