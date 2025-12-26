import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import Cache

from model.wrap_hf import create_model_class, create_config_class

from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from accelerate import init_empty_weights

class StreamingLLMAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.sliding_window = config.sliding_window

    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,

        use_cache: bool,
        hidden_states: torch.Tensor,
        position_embeddings,
        past_key_values,
        cache_position,
        **kwargs
    ):
        q, k, v = query, key, value
        del query, key, value
        B, H, L, N = q.shape
        B, KVH, S, N = k.shape
        
        sdpa_kwargs = {}

        if H > KVH:
            if attention_mask is None:
                sdpa_kwargs = {"enable_gqa": True}
            else:
                k = repeat_kv(k, H // KVH)
                v = repeat_kv(v, H // KVH)

        sliding_window = self.sliding_window
        if sliding_window > 0:
            q_idx = torch.arange(S-L, S, device=q.device)[None, None, :, None]
            kv_idx = torch.arange(S, device=q.device)[None, None, None, :]
            window_mask = kv_idx >= q_idx - sliding_window
            if attention_mask is not None:
                assert attention_mask.dtype == torch.bool
                sink_indices = (S - attention_mask.view(B,L,S)[:,-1,:].sum(dim=-1)).view(B) # sink offset per batch idx
                sink_mask = kv_idx == sink_indices.view(B,1,1,1)
                #prefill_mha_mask = q_idx >= S - prefill_n_mha_tokens
                #attention_mask = attention_mask & (sink_mask | window_mask | prefill_mha_mask)
                attention_mask = attention_mask & (sink_mask | window_mask)
            else:
                sink_mask = kv_idx == 0
                attention_mask = kv_idx <= q_idx # causal
                attention_mask = attention_mask & (sink_mask | window_mask)

        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, :S]

        is_causal = L > 1 and attention_mask is None
        if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
            is_causal = is_causal.item()

        attn_output = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attention_mask, dropout_p=dropout, is_causal=is_causal, scale=scaling, **sdpa_kwargs)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers import AutoConfig, PretrainedConfig
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import datasets
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
import os

@dataclass
class TokenizingCollator:
    tokenizer: PreTrainedTokenizer
    max_length: int

    def __call__(self, examples: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, torch.Tensor]:
        texts = [example['text'] for example in examples]
        #print('[len(text) for text in texts]', [len(text) for text in texts])
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length + 1, # FIXME - fixed this from max_length without one added
            padding="max_length",
            return_tensors="pt",
            padding_side='right'
        )

        input_ids = tokenized["input_ids"][:, :-1]
        attention_mask = tokenized["attention_mask"][:, :-1]
        labels = tokenized["input_ids"][:, 1:].clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def worker_process(local_rank:int, world_size:int, *args, **kwargs):
    try:
        if world_size > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=local_rank,
                device_id=local_rank,
            )

        _worker_process(local_rank, world_size, *args, **kwargs)
    except Exception as e:
        if local_rank == 0:
            #print(f"Error in worker: {e}")
            import traceback
            print(f"Error in worker\n", traceback.format_exc())

    if world_size > 1:      
        dist.barrier()
        if local_rank == 0:
            print("Tearing down process group...")
        dist.destroy_process_group()

    if local_rank == 0:
        print("Done!")

@dataclass(kw_only=True)
class CLI_Config:
    num_gpus:int|None = None
    ctxlen:int = 2048
    micro_bsz:int = 32
    max_iters:int = 1
    dataset_name:str = "robbiegwaldd/dclm-10B"
    model_path:str = 'Qwen/Qwen2-3B-Instruct'
    base_model_class_path:str = 'transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM'
    base_attention_class_path:str = 'transformers.models.qwen2.modeling_qwen2.Qwen2Attention'
    base_config_class_path:str = 'transformers.models.qwen2.configuration_qwen2.Qwen2Config'
    sliding_window_size:int = 256
    full_attention_layer_ids:list = field(default_factory=list)
    seed:int = 1337
    iterate:int = 1


def _worker_process(local_rank:int, world_size:int, cli_config:CLI_Config):
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(local_rank)
    #torch.set_default_device(device)

    tokenizer = AutoTokenizer.from_pretrained(cli_config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    StreamingLLMHybridForCausalLM = create_model_class(
        StreamingLLMAttention, 
        cli_config.base_model_class_path,
        cli_config.base_attention_class_path,
    )

    StreamingLLMHybridConfig = create_config_class(cli_config.base_config_class_path)

    if local_rank == 0: print("loading config", cli_config.model_path)
    config_dict, unused_kwargs = PretrainedConfig.get_config_dict(cli_config.model_path, _from_auto=True)
    model_config = StreamingLLMHybridConfig.from_dict(config_dict, **unused_kwargs)

    # NOTE - entirely replacement attentions, and we will change the sliding window size as needed to simulate the original model
    model_config.layer_hybrid_types = ['radlads_replacement_attention'] * model_config.num_hidden_layers

    if local_rank == 0: print("instantiating customized model", cli_config.model_path)
    with init_empty_weights():
        model = StreamingLLMHybridForCausalLM(model_config)

    if local_rank == 0: print("loading original model weights", cli_config.model_path)
    base_model = AutoModelForCausalLM.from_pretrained(cli_config.model_path, device_map=device)
    base_weights = base_model.state_dict()
    del base_model

    if local_rank == 0: print("moving original model weights", cli_config.model_path)
    model.load_state_dict(base_weights, assign=True)
    del base_weights

    model.eval()

    dataset = datasets.load_dataset(cli_config.dataset_name)['train'] #, streaming=True)

    if local_rank == 0: print(f"model: {cli_config.model_path} dataset: {cli_config.dataset_name}")
    if local_rank == 0: print(f"layer_id,kl_div_loss")

    sampler = None
    # if world_size > 1:
    #     sampler = DistributedSampler(
    #         dataset=dataset,
    #         num_replicas=world_size,
    #         rank=local_rank,
    #         shuffle=False, # NOTE - no shuffling
    #     )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cli_config.micro_bsz,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=TokenizingCollator(tokenizer, cli_config.ctxlen),
        sampler=sampler,
        shuffle=True,
    )

    with torch.no_grad():
        layer_count = len(model.model.layers)
        for layer_id in range(local_rank, layer_count, world_size if cli_config.iterate else 999999):
            if cli_config.iterate and layer_id in cli_config.full_attention_layer_ids:
                continue
            total_loss = torch.zeros([1], device=device)
            total_batchlen = 0
            torch.manual_seed(cli_config.seed)
            for step, data in enumerate(dataloader):
                if step >= cli_config.max_iters:
                    break

                input_ids = data['input_ids'].to(device)
                # labels = data['labels'].to(device)
                attention_mask = data['attention_mask'].to(device=device, dtype=torch.bool)

                # change to teacher model with all full attention (no swa window)
                for layer_id2 in range(model_config.num_hidden_layers):
                    model.model.layers[layer_id2].self_attn.attn_replacement.sliding_window = 0

                # run teacher model
                teacher_logits = model(input_ids).logits

                # change to student model with a single additional attention layer
                for layer_id2 in range(model_config.num_hidden_layers):
                    model.model.layers[layer_id2].self_attn.attn_replacement.sliding_window = cli_config.sliding_window_size
                for layer_id2 in cli_config.full_attention_layer_ids:
                    model.model.layers[layer_id2].self_attn.attn_replacement.sliding_window = 0
                if cli_config.iterate:
                    model.model.layers[layer_id].self_attn.attn_replacement.sliding_window = 0

                # run student model
                student_logits = model(input_ids).logits

                student_logits = student_logits.masked_fill_(~attention_mask.unsqueeze(-1), -9999999)
                teacher_logits = teacher_logits.masked_fill_(~attention_mask.unsqueeze(-1), -9999999)
               
                flat_attention_mask = attention_mask.view(-1)
                flat_student_logits = student_logits.view(-1, student_logits.size(-1))[flat_attention_mask]
                flat_teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))[flat_attention_mask]
                # flat_labels = labels.view(-1)[flat_attention_mask]
                # print('teacher ce', F.cross_entropy(flat_teacher_logits, flat_labels))
                # print('student ce', F.cross_entropy(flat_student_logits, flat_labels))

                #if local_rank == 0:
                #    print(f"Layer {layer_id} step {step} len {flat_student_logits.size(0)}")
                chunk_size = 256
                for i in range(0, flat_student_logits.shape[0], 256):
                    total_loss += F.kl_div(F.log_softmax(flat_student_logits[i:i+chunk_size], dim=-1), F.log_softmax(flat_teacher_logits[i:i+chunk_size], dim=-1), reduction='sum', log_target=True)
                total_batchlen += flat_student_logits.size(0)

            total_loss /= total_batchlen
            
            #if world_size > 1:
            #    dist.reduce(total_loss, 0, op = ReduceOp.AVG)
            # FIXME - save result
            print(f"{layer_id},{total_loss.item()}")


if __name__ == '__main__':
    import sys
    from config import parse_cmdline_configs
    cli_config, errors = parse_cmdline_configs(sys.argv[1:], CLI_Config)
    if errors != '':
        print(errors)
        exit(-1)

    if cli_config.num_gpus == 1:
        worker_process(0, 1, cli_config)
    else:
        # Required for CUDA multiprocessing
        mp.set_start_method('spawn', force=True) 

        if cli_config.num_gpus is None or cli_config.num_gpus == 0:
            cli_config.num_gpus = torch.cuda.device_count()

        if cli_config.num_gpus == 0:
            raise RuntimeError("No GPUs available!")

        # Use multiprocessing Manager for sharing results
        manager = mp.Manager()

        # Spawn processes for each GPU
        mp.spawn(
            worker_process,
            args=[cli_config.num_gpus, cli_config, ],
            nprocs=cli_config.num_gpus,
            join=False, #True
        )
