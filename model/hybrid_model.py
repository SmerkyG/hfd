import sys
import os
from typing import Optional, Tuple
import torch._dynamo

# たとえば、再コンパイルの上限を256回に設定
#torch._dynamo.config.recompile_limit = 256
from torch.utils.checkpoint import checkpoint as torch_checkpoint

RWKV_VERSION=os.environ.get('RWKV_VERSION','v7')
is_rwkv_7 = RWKV_VERSION == 'v7'
#if is_rwkv_7 :
if os.environ["architecture"] == 'hxa07c':
    from .TimeMixer import RWKV_Tmix_x070_Mose_cxa07C as TimeMixer
from .TimeMixer import GQAWithRopeAttention as SelfAttention

import torch
from torch.nn import functional as F
import torch.nn as nn

import gc
import logging
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

#global
current_embeddings = None
def embedding_hook(module,input,output):
    global current_embeddings
    if isinstance(module,nn.Embedding):
        #print(f'embedding detected = {output.shape}')
        current_embeddings = output

class AttentionWrapper(nn.Module):
    
    def __init__(self,student_attn,layer_idx,args):
        super(AttentionWrapper, self).__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.teacher_attn = None
        self.student_attn = student_attn
        if self.student_attn is not None:
            self.student_attn.requires_grad_(True)
            print(f'Layer:{layer_idx} is trained')
        else:
            print(f'Layer:{layer_idx} is not trained')
        self.add_module("student_attn", self.student_attn)
        self.global_rank = None
        self.attention_mask = None


    def forward(self, 
        # hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Cache] = None,
        # output_attentions: bool = False,
        # use_cache: bool = False,
        # cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        attention_hidden_states,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #kwargs['output_attentions'] = False

        # NOTE - instead of returning attentions here we return a special attention loss
        hidden_states = kwargs['hidden_states']
        position_embeddings = kwargs['position_embeddings']
        position_ids = kwargs['position_ids']
        attention_mask = kwargs['attention_mask']

        global current_embeddings
        #print(f'{(attention_mask.shape)}')

        if self.student_attn is not None:

            hidden_states = hidden_states.requires_grad_(True)
            # print(f"AttentionWrapper: layer_idx={self.layer_idx}, attention_mask={self.attention_mask}")
            # print(f"kargs={kwargs}")
            if self.args.grad_cp == 1:
                if is_rwkv_7:
                    if self.student_attn.Attention:
                        if self.args.freeze_hybrid_attention:
                            with torch.no_grad():
                                if self.args.stage == 2:
                                    student_hidden_states = self.student_attn(hidden_states, position_embeddings,current_embeddings)
                                else:
                                    teacher_outputs = self.teacher_attn(*args, **kwargs)
                                    student_hidden_states = teacher_outputs[0]
                        else:
                            student_hidden_states = torch_checkpoint(self.student_attn, hidden_states, position_embeddings,current_embeddings, use_reentrant=False)
                    else:
                        student_hidden_states = torch_checkpoint(self.student_attn, hidden_states, self.attention_mask, position_embeddings,position_ids, current_embeddings,use_reentrant=False)
                # else:
                #     # we not using
                #    # student_hidden_states = deepspeed.checkpointing.checkpoint(self.student_attn, hidden_states)
                #    student_hidden_states = torch_checkpoint(self.student_attn, hidden_states, position_embeddings, use_reentrant=False)
    
            
            #print(student_hidden_states)
            if self.args.stage != 1:
                return (student_hidden_states, None)
            # student_outputs = self.student_attn(hidden_states)


        if self.student_attn.Attention and self.args.freeze_hybrid_attention:
                teacher_outputs = teacher_outputs
        else:
            with torch.no_grad():
                teacher_outputs = self.teacher_attn(*args, **kwargs)
        # special attention loss is the vector norm of the difference between the student and teacher attn outputs
        # student_hidden_states = student_outputs[0]
        teacher_hidden_states = teacher_outputs[0]


  
        if self.student_attn is None:
            #special_attn_loss = 0.0
            pass
        else:
            #special_attn_loss = self.comprehensive_attention_mimicking_loss_improved(teacher_hidden_states,student_hidden_states,self.layer_idx,self.args.n_layer,self.args)
            #special_attn_loss = torch.nn.functional.mse_loss(student_hidden_states, teacher_hidden_states)

            #special_attn_loss = (student_hidden_states, teacher_hidden_states)
            if attention_hidden_states is not None:
                attention_hidden_states += [(student_hidden_states, teacher_hidden_states)]

       # print('Will Return Attention Output')

       # FIXME - instead of returning as attention outputs, return as hidden states! (and use SDPA for teacher instead of eager by having calling code not ask for attention_outputs)

        #return (teacher_outputs[0], special_attn_loss ) + teacher_outputs[2:]
        return teacher_outputs

class HybridModel(nn.Module):
    
    @staticmethod
    def get_rwkv_args(transformer_config):
        from argparse import Namespace
        args = Namespace()
        args.my_pos_emb = 0
        args.head_size_a = 64
        args.head_size_divisor = 8
        args.ctx_len = 4096
        args.n_layer = transformer_config.num_hidden_layers
        args.n_embd = transformer_config.hidden_size
        args.dim_att = transformer_config.hidden_size
        args.dim_ffn = transformer_config.intermediate_size
        args.pre_ffn = 0
        args.head_qk = 0
        args.tiny_att_dim = 0
        args.tiny_att_layer = -999
        args.vocab_size = transformer_config.vocab_size
        args.layers = [i for i in range(transformer_config.num_hidden_layers)]
        args.pad_id = transformer_config.eos_token_id
        args.stage = 4
        args.is_rwkv_att_only = True
        args.is_all_labels_kl = True
        args.init_with_llama = False
        return args
    
    def __init__(self, transformer_model, rwkv_args, tokenizer=None):
        super(HybridModel, self).__init__()
        stage = rwkv_args.stage
        if stage == 1:
            #Freeze the model
            transformer_model.requires_grad_(False)
        else:
            # for stage2 freeze transformer model
            #transformer_model.requires_grad_(False)
            # only train attention
            ##Unfreeze the model
            transformer_model.requires_grad_(True)
            if transformer_model.config.tie_word_embeddings:
                # copy untied embeddings
                transformer_model.get_output_embeddings().weight = nn.Parameter(transformer_model.get_input_embeddings().weight.clone())
                # untie the embeddings in the config, too
                transformer_model.tie_word_embeddings = False
        self.args = rwkv_args

        for layer_idx in range(transformer_model.config.num_hidden_layers):
            if layer_idx in rwkv_args.layers:
                if layer_idx in self.args.transformer_layers:
                    print(f'layer:{layer_idx} Attention Layer')
                    student_attn = SelfAttention(rwkv_args, layer_idx)
                else:
                    print(f'layer:{layer_idx} RWKV Layer')
                    student_attn = TimeMixer(rwkv_args, layer_idx)

                # if layer_idx < (transformer_model.config.num_hidden_layers - self.args.hybrid_attention_layers):
                #     student_attn = TimeMixer(rwkv_args, layer_idx)
                # else:
                    



                llama_layer = transformer_model.model.layers[layer_idx]
                attn_wrapper = AttentionWrapper(student_attn,layer_idx,rwkv_args)
                llama_layer.self_attn = attn_wrapper
                gc.collect()

        #exit()
        self.model = transformer_model
        self.add_module("model", self.model)

        self.model.get_input_embeddings().register_forward_hook(embedding_hook)
        
        self.teacher_model = None  # 初始化为None，后续再设置
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            if 'pad_token_id' not in self.tokenizer.__dict__:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        torch.cuda.empty_cache()
        self.client = None

    #@torch.compile
    def forward(
        self,
        input_ids,
        attention_mask,
        **kwargs,
    ):
        for layer_idx in range(self.model.config.num_hidden_layers):
            if layer_idx in self.args.layers:
                llama_layer = self.model.model.layers[layer_idx]
                attn_wrapper = llama_layer.self_attn
                attn_wrapper.attention_mask = attention_mask
        ret = self.model(input_ids, **kwargs)
        return ret
    
    def load_check_point(self, path):
        all_keys = set(self.state_dict().keys())
        incompatible_keys = set()
        #if the path is the file, load it directly
        #if the path is the directory, load the sharded files in the directory with suffix .pt or .bin
        if os.path.isdir(path):
            files = os.listdir(path)
            files = [os.path.join(path, f) for f in files if f.endswith('.pt') or f.endswith('.bin')]
        else:
            files = [path]
        for file in files:
            checkpoint = torch.load(file, map_location='cpu')
            self.load_state_dict(checkpoint, strict=False)
            print(f'load model from {file}')
            ckpt_keys = checkpoint.keys()
            #subtract the keys in the checkpoint from the all_keys
            #if the ckpt_key exists in the all_keys, remove it
            for ckpt_key in ckpt_keys:
                if ckpt_key in all_keys:
                    all_keys.remove(ckpt_key)
                else:
                    incompatible_keys.add(ckpt_key)
            del checkpoint
            gc.collect()
        print(f'Finish loading model from {path}')
        print(f'Incompatible keys: {incompatible_keys} missing keys: {all_keys}')
        
        
        return
