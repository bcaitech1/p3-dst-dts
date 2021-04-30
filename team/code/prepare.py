from attrdict import AttrDict
from importlib import import_module

import torch
from transformers import AutoTokenizer
from data_utils import get_examples_from_dialogues

def get_stuff(args, train_data, dev_data, slot_meta, ontology):
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
        train_data, user_first=user_first, dialogue_level=dialogue_level
    )

    if dev_data is not None:
        dev_examples = get_examples_from_dialogues(
            dev_data, user_first=user_first, dialogue_level=dialogue_level
        )

    # Define Preprocessor
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    processor = getattr(import_module('preprocessor'), args.preprocessor)(
        slot_meta, tokenizer, **processor_kwargs
    )
    args.vocab_size = len(tokenizer)

    if args.preprocessor == 'TRADEPreprocessor':
        args.n_gate = len(processor.gating2id) # gating 갯수 none, dontcare, ptr

    # Extracting Featrues
    print('Converting examples to features')
    train_features = processor.convert_examples_to_features(train_examples)
    if dev_data is not None:
        dev_features = processor.convert_examples_to_features(dev_examples)
    else:
        dev_features = None
    print('Done converting examples to features')
    
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
    if args.model_class == 'TRADE':
        tokenized_slot_meta = []
        for slot in slot_meta:
            tokenized_slot_meta.append(
                tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
            )

        model_kwargs = AttrDict(
            tokenized_slot_meta=tokenized_slot_meta,
        )
    elif args.model_class == 'SUMBT':
        slot_type_ids, slot_values_ids = tokenize_ontology(ontology, tokenizer, args.max_label_length)
        num_labels = [len(s) for s in slot_values_ids]

        model_kwargs = AttrDict(
            num_labels=num_labels,
            device=args.device,
        )
    else:
        raise NotImplementedError()

    model = getattr(import_module('model'), args.model_class)(
        args, **model_kwargs
    )

    if args.model_class == 'TRADE':
        model.set_subword_embedding(args.model_name_or_path)  # Subword Embedding 초기화    
    elif args.model_class == 'SUMBT':
        model.initialize_slot_value_lookup(slot_values_ids, slot_type_ids)  # Tokenized Ontology의 Pre-encoding using BERT_SV        

    return model