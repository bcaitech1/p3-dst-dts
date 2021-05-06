from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_position_embedding import PositionEmbedding 

from .sumbt import AutoForUtteranceEncoding, MultiHeadAttention, Transformer

class SUMBT_Gen(nn.Module):
    def __init__(self, args, num_slots, device, pad_idx=0):
        super().__init__()

        self.hidden_dim = args.hidden_dim
        self.rnn_num_layers = args.num_rnn_layers
        self.zero_init_rnn = args.zero_init_rnn
        self.max_seq_length = args.max_seq_length
        self.max_label_length = args.max_label_length
        self.num_slots = num_slots
        self.attn_head = args.attn_head
        self.use_larger_slot_encoding = args.use_larger_slot_encoding
        self.use_transformer = args.use_transformer
        self.device = device
        self.n_gate = args.n_gate

        ### Utterance Encoder
        self.utterance_encoder = AutoForUtteranceEncoding.from_pretrained(
            args.model_name_or_path
        )
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        try:
            self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        except AttributeError:
            self.hidden_dropout_prob = 0.1

        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.base.pooler.parameters():
                p.requires_grad = False

        ### slot, slot-value Encoder (not trainable)
        self.sv_encoder = AutoForUtteranceEncoding.from_pretrained(
            args.model_name_or_path
        )
        # os.path.join(args.bert_dir, 'bert-base-uncased.model'))
        for p in self.sv_encoder.base.parameters():
            p.requires_grad = False

        new_vocab_size = self.utterance_encoder.base.bert.embeddings.word_embeddings.num_embeddings + 1
        self.utterance_encoder.base.resize_token_embeddings(new_vocab_size)

        self.vocab_size = new_vocab_size

        if self.use_larger_slot_encoding:
            self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim*self.max_label_length)
            self.slot_pooler = nn.Linear(self.max_label_length * self.bert_output_dim, self.bert_output_dim)
        else:
            self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim)

        ### Attention layer
        self.attn = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0)
        
        if self.use_transformer:
            self.transformers = nn.ModuleList(
                [Transformer(self.attn_head, self.bert_output_dim, dropout=0) for _ in range(args.n_transformers)]
            )

            self.position_emb = PositionEmbedding(num_embeddings=50,
                embedding_dim=self.bert_output_dim,
                mode=PositionEmbedding.MODE_ADD)
        else:
            if self.rnn_num_layers == 1:
                gru_dropout_prob = 0
            else:
                gru_dropout_prob = self.hidden_dropout_prob
            self.nbt = nn.GRU(
                input_size=self.bert_output_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.rnn_num_layers,
                dropout=gru_dropout_prob,
                batch_first=True,
            )
            self.init_parameter(self.nbt)

            if not self.zero_init_rnn:
                self.rnn_init_linear = nn.Sequential(
                    nn.Linear(self.bert_output_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.hidden_dropout_prob),
                )

            self.linear = nn.Linear(self.hidden_dim, self.bert_output_dim)
            self.layer_norm = nn.LayerNorm(self.bert_output_dim)

        ### Measure
        # self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        ### Classifier
        # self.nll = CrossEntropyLoss(ignore_index=-1)

        ### Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.bos_idx = new_vocab_size - 1

        self.decoder_gru = nn.GRU(
            self.hidden_dim, self.hidden_dim, num_layers=1, dropout=0, batch_first=True
        )

        self.w_gen = nn.Linear(self.hidden_dim * 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.w_gate = nn.Linear(self.hidden_dim, self.n_gate)
        
        self.vocab_embed = self.utterance_encoder.base.bert.embeddings.word_embeddings

    def embedding(self, x):
        x = self.vocab_embed(x)
        # if self.proj_layer:
        #     x = self.proj_layer(x)
        return x

    def initialize_slot_value_lookup(self, slot_ids):
        self.sv_encoder.eval()

        # Slot encoding
        pbar = tqdm(desc='Making slot encoding -- waiting...', bar_format='{desc} -> {elapsed}')
        slot_type_ids = torch.zeros(slot_ids.size(), dtype=torch.long).to(
            slot_ids.device
        )
        slot_mask = slot_ids > 0
        hid_slot = self.sv_encoder(
            slot_ids.view(-1, self.max_label_length),
            slot_type_ids.view(-1, self.max_label_length),
            slot_mask.view(-1, self.max_label_length),
        ).hidden_states[-1]
        
        if self.use_larger_slot_encoding:
            hid_slot = hid_slot.masked_fill(slot_mask.unsqueeze(-1) == False, 0)
            hid_slot = hid_slot.view(self.num_slots, -1)
        else:
            hid_slot = hid_slot[:, 0, :]
        hid_slot = hid_slot.detach()
    
        old = self.slot_lookup
        self.slot_lookup = nn.Embedding.from_pretrained(hid_slot, freeze=True)
        assert self.slot_lookup.weight.shape == old.weight.shape, f'{self.slot_lookup.weight.shape} {old.weight.shape}'
        pbar.set_description('Making slot encoding -- DONE')
        pbar.close()

        self.sv_encoder = None

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        max_len,
        target_slot=None,
        teacher=None
    ):
        # B = Batch Size
        # M = Max Turn Size
        # N = Seq Len
        # J = Target_slot Len
        # H_GRU = RNN Hidden Dim
        # L = Label Len
        # V = Vocab Size
        # G = Generated Output Len(max_len)
        
        # input_ids: [B, M, N]
        # token_type_ids: [B, M, N]
        # attention_mask: [B, M, N]
        # teacher: [B, M, J, G]

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0)  # Batch size (B)
        ts = input_ids.size(1)  # Max turn size (M)
        bs = ds * ts # B * M
        slot_dim = len(target_slot)  # J

        # Utterance encoding
        hidden = self.utterance_encoder(
            input_ids.view(-1, self.max_seq_length),
            token_type_ids.view(-1, self.max_seq_length),
            attention_mask.view(-1, self.max_seq_length),
        ).hidden_states[-1] # [B*M, N, H]

        hidden = torch.mul(
            hidden,
            attention_mask.view(-1, self.max_seq_length, 1)
            .expand(hidden.size())
            .float(),
        )
        hidden = hidden.repeat(slot_dim, 1, 1)  # [J*B*M, N, H]
        encoder_output = hidden # [J*B*M, N, H]

        hid_slot = self.slot_lookup.weight[
            target_slot, :
        ]  # Select target slot embedding

        if self.use_larger_slot_encoding: # [J, H * L]
            hid_slot = hid_slot.view(slot_dim, self.max_label_length, self.bert_output_dim) # [J, L, H]
            hid_slot = hid_slot.repeat(1, bs, 1).view(bs * slot_dim, self.max_label_length, -1) # [J*M*B, L, H]
        else: # [J, H]
            hid_slot = hid_slot.repeat(1, bs).view(bs * slot_dim, -1)  # [J*M*B, H]

        # Attended utterance vector
        hidden = self.attn(
            hid_slot,  # q^s  [J*M*B, L, H]
            hidden,  # U [J*M*B, N, H]
            hidden,  # U [J*M*B, N, H]
            mask=attention_mask.view(-1, 1, self.max_seq_length).repeat(slot_dim, 1, 1),
        ) # [J*M*B, L, H]

        if self.use_larger_slot_encoding: # [J*M*B, H]
            hidden = self.slot_pooler(hidden.view(-1, self.max_label_length * self.bert_output_dim))
        else:
            hidden = hidden.squeeze()  # h [J*M*B, H] Aggregated Slot Context

        hidden = hidden.view(slot_dim, ds, ts, -1).view(
            -1, ts, self.bert_output_dim
        )  # [J*B, M, H]

        if self.use_transformer:
            hidden = self.position_emb(hidden)

            transformer_mask = torch.any(attention_mask, -1).repeat(slot_dim, 1).unsqueeze(1)
            for transformer in self.transformers:
                hidden = transformer(
                    hidden,
                    mask=transformer_mask # [J*B, 1, M]
                ) # [J*B, M, H]
        else:
            # NBT
            if self.zero_init_rnn:
                h = torch.zeros(
                    self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim
                ).to(
                    self.device
                )  # [1, slot_dim*ds, hidden]
            else:
                h = hidden[:, 0, :].unsqueeze(0).expand(self.rnn_num_layers, -1, -1) # 원래 repeat(self.rnn_num_layers, 1, 1) 이었는데 바꿈
                h = self.rnn_init_linear(h)    # 왜냐면 1 dim은 언제나 사이즈 1이니까 expand 사용하는게 더 좋음(expand 메모리 복사 없음, repeat 복사됨)

            if isinstance(self.nbt, nn.GRU):
                rnn_out, _ = self.nbt(hidden, h)  # [J*B, M, H_GRU]
            elif isinstance(self.nbt, nn.LSTM):
                c = torch.zeros(
                    self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim
                ).to(
                    self.device
                )  # [1, slot_dim*ds, hidden]
                rnn_out, _ = self.nbt(hidden, (h, c))  # [slot_dim*ds, turn, hidden]
            hidden = self.layer_norm(self.linear(self.dropout(rnn_out)))

        hidden = hidden.view(slot_dim, ds, ts, -1) # [J, B, M, H]

        hidden = hidden.view(slot_dim * ds * ts, -1) # [J*B*M, H]

        w = torch.ones(slot_dim * ds * ts, dtype=torch.long, device=self.device)* self.bos_idx # [J*B*M]
        w = self.embedding(w) # [J*B*M, H]

        hidden = hidden.unsqueeze(0) # [1, J*B*M, H]
        w = w.unsqueeze(1) # [J*B*M, 1, H]

        attention_mask = attention_mask.ne(1) # [B, M, N]
        attention_mask = attention_mask.view(ds * ts, -1).repeat(slot_dim, 1) # [J*B*M, N]

        input_ids = input_ids.view(ds * ts, -1).repeat(slot_dim, 1) # [J*B*M, N]

        all_point_outputs = torch.zeros(slot_dim, ds, ts, max_len, self.vocab_size,
         device=input_ids.device) # [J, B, M, G, V]

        for k in range(max_len):
            w = self.dropout(w)
            _, hidden = self.decoder_gru(w, hidden) # [1, J*B*M, H]

            # [J*B*M, N, H] * [J*B*M, H, 1] = [J*B*M, N, 1]
            attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # [J*B*M, N, 1]
            attn_e = attn_e.squeeze(-1).masked_fill(attention_mask, -10000.0) # [J*B*M, N]
            attn_history = F.softmax(attn_e, -1)  # [J*B*M, N]

            # [J*B*M, H] * [H, V] = [J*B*M, V]
            attn_v = torch.matmul(
                hidden.squeeze(0), self.vocab_embed.weight.transpose(0, 1)
            )  # [J*B*M, V]
            attn_vocab = F.softmax(attn_v, -1) # [J*B*M, N]

            # [J*B*M, 1, N] * [J*B*M, N, H] = [J*B*M, 1, H]
            context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # [J*B*M, 1, H]

            # w [J*B*M, 1, H]
            # hidden.T [J*B*M, 1, H]
            # context [J*B*M, 1, H]
            # cat [J*B*M, 3*H]
            p_gen = self.sigmoid(
                self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1))
            )  # [J*B*M, 1]
            p_gen = p_gen.squeeze(-1) # [J*B*M]

            # input_ids: [J*B*M, N]
            # atten_vocab: [J*B*M, N]
            p_context_ptr = torch.zeros_like(attn_vocab).to(input_ids.device) # [J*B*M, V]
            p_context_ptr.scatter_add_(1, input_ids, attn_history)  # [J*B*M, V]
            p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # [J*B*M, V]
            _, w_idx = p_final.max(-1) # [J*B*M]
            
            # teacher[:,:,:,k] [B, M, J]
            if teacher is not None: # [B, M, J, G]
                cur_teacher = teacher[:, :, :, k] # [B, M, J]
                cur_teacher = cur_teacher.permute(2, 0, 1) # [J, B, M]
                cur_teacher = cur_teacher.reshape(slot_dim*ds*ts, -1) # [J*B*M, 1]
                w = self.embedding(cur_teacher) # [J*B*M, 1, H]
            else:
                w = self.embedding(w_idx).unsqueeze(1)  # [J*B*M, 1, H]
            if k == 0:
                # context [J*B*M, 1, H]
                gated_logit = self.w_gate(context.squeeze(1))  # [J*B*M, n_gate]
                all_gate_outputs = gated_logit.view(slot_dim, ds, ts, self.n_gate) # [J, B, M, n_gate]
            
            all_point_outputs[:, :, :, k, :] = p_final.view(slot_dim, ds, ts, self.vocab_size)
            # [J, B, M, G, V]

        all_point_outputs = all_point_outputs.permute(1, 2, 0, 3, 4) # [B, M, J, G, V]
        all_gate_outputs = all_gate_outputs.permute(1, 2, 0, 3) # [B, M, J, n_gate]
        return all_point_outputs, all_gate_outputs 
        
    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)


