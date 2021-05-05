from attrdict import AttrDict
from importlib import import_module
import json

from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer
from data_utils import get_examples_from_dialogues

gen_slot_meta = set(['관광-이름', '숙소-이름', '식당-이름', '택시-도착지', '택시-출발지'])
gen_domain = set(s.split('-')[0] for s in gen_slot_meta)

def get_data(args):
    train_data_file = f"{args.data_dir}/train_dials.json"
    data = json.load(open(train_data_file))
    if args.use_small_data:
        data = data[:100]
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
    ontology = json.load(open(f"{args.data_dir}/ontology.json"))

    if args.use_generation_only:
        print(f'Using only {" ".join(gen_slot_meta)}')
        old_data = data
        data = []
        for dial in old_data:
            if any([x in gen_domain for x in dial['domains']]):
                data.append(dial)
        slot_meta = list(gen_slot_meta)
        ontology = {k:v for k, v in ontology.items() if k in gen_slot_meta}

    return data, slot_meta, ontology

def filter_examples(examples, filter_slot_meta, which, dev_labels=None):
    filtered_examples = []
    print(f'Filtering {which} dialogues without slot meta: {" ".join(filter_slot_meta)}')
    old_len = len(examples)
    pbar = tqdm(enumerate(examples), desc=f'Filtering {which} dialogues by slot meta',
                total=old_len)
    for idx, example in pbar:
        have = False
        if dev_labels is not None:
            using_labels = dev_labels[example.guid]
        else:
            using_labels = example.label

        for x in using_labels:
            domain, slot, value = x.split('-')
            if f'{domain}-{slot}' in filter_slot_meta:
                have = True
                break
        if have:
            filtered_examples.append(example)
    pbar.close()

    new_len = len(filtered_examples)
    print(f'Filtered {which} results: {old_len} -> {new_len}')
    return filtered_examples

def get_stuff(args, train_data, dev_data, slot_meta, ontology, dev_labels=None):
    if args.preprocessor == 'TRADEPreprocessor':
        user_first = False
        dialogue_level = False
        processor_kwargs = AttrDict()
    elif args.preprocessor == 'SUMBTPreprocessor':
        user_first = True
        dialogue_level = True
        max_turn = max([len(e['dialogue']) for e in train_data])
        processor_kwargs = AttrDict(
            ontology=ontology,
            max_turn_length=max_turn,
            max_seq_length=args.max_seq_length,
            model_name_or_path=args.model_name_or_path,
        )
    else:
        raise NotImplementedError()

    train_examples = get_examples_from_dialogues(
        train_data, user_first=user_first, dialogue_level=dialogue_level, which='train'
    )

    if dev_data is not None:
        dev_examples = get_examples_from_dialogues(
            dev_data, user_first=user_first, dialogue_level=dialogue_level, which='val'
        )

    if args.use_gen_dialog_only:
        if args.ModelName == 'SUMBT':
            raise NotImplementedError('SUMBT는 구현 안함')
        train_examples = filter_examples(train_examples, gen_slot_meta, which='train')
        if dev_data is not None:
            dev_examples = filter_examples(dev_examples, gen_slot_meta, which='val', dev_labels=dev_labels)

    # Define Preprocessor
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    processor = getattr(import_module('preprocessor'), args.preprocessor)(
        slot_meta, tokenizer, **processor_kwargs
    )
    args.vocab_size = len(tokenizer)

    if args.preprocessor == 'TRADEPreprocessor':
        args.n_gate = len(processor.gating2id) # gating 갯수 none, dontcare, ptr

    # Extracting Featrues
    # print('Converting examples to features')
    train_features = processor.convert_examples_to_features(train_examples, which='train')
    if dev_data is not None:
        dev_features = processor.convert_examples_to_features(dev_examples, which='val')
    else:
        dev_features = None
    # print('Done converting examples to features')
    
    return tokenizer, processor, train_features, dev_features


def tokenize_ontology(ontology, tokenizer, max_seq_length):
    slot_types = []
    slot_values = []
    for k, v in ontology.items():
        tokens = tokenizer.encode(k)
        if len(tokens) < max_seq_length:
            gap = max_seq_length - len(tokens)
            tokens.extend([tokenizer.pad_token_id] * gap)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        slot_types.append(tokens)
        slot_value = []
        for vv in v:
            tokens = tokenizer.encode(vv)
            if len(tokens) < max_seq_length:
                gap = max_seq_length - len(tokens)
                tokens.extend([tokenizer.pad_token_id] * gap)
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
            slot_value.append(tokens)
        slot_values.append(torch.LongTensor(slot_value))
    return torch.LongTensor(slot_types), slot_values

def get_model(args, tokenizer, ontology, slot_meta):
    if args.ModelName == 'TRADE':
        tokenized_slot_meta = []
        for slot in slot_meta:
            tokenized_slot_meta.append(
                tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
            )

        model_kwargs = AttrDict(
            tokenized_slot_meta=tokenized_slot_meta,
        )
    elif args.ModelName == 'SUMBT':
        slot_type_ids, slot_values_ids = tokenize_ontology(ontology, tokenizer, args.max_label_length)
        num_labels = [len(s) for s in slot_values_ids]

        model_kwargs = AttrDict(
            num_labels=num_labels,
            device=args.device,
        )
    else:
        raise NotImplementedError()

    pbar = tqdm(desc=f'Making {args.model_class} model -- waiting...', bar_format='{desc} -> {elapsed}')
    model = getattr(import_module('model'), args.model_class)(
        args, **model_kwargs
    )
    pbar.set_description(f'Making {args.model_class} model -- DONE')    
    pbar.close()

    if args.ModelName == 'TRADE':
        pbar = tqdm(desc='Setting subword embedding -- waiting...', bar_format='{desc} -> {elapsed}')
        model.set_subword_embedding(args.model_name_or_path)  # Subword Embedding 초기화    
        pbar.set_description('Setting subword embedding -- DONE')
        pbar.close()
    elif args.ModelName == 'SUMBT':
        print('Initializing slot value lookup --------------')
        model.initialize_slot_value_lookup(slot_values_ids, slot_type_ids)  # Tokenized Ontology의 Pre-encoding using BERT_SV        
        print('Finished initializing slot value lookup -----')

    return model