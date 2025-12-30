from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.nn import functional as F





from transformers import Cache, PretrainedConfig
from transformers.cache_utils import DynamicLayer

class CompressingCache(Cache):
    def __init__(self, *args, **kwargs):
        super().__init__(layer_class_to_replicate=CompressingLayer, *args, **kwargs)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_kwargs['layer_idx'] = layer_idx

        return super().update(key_states, value_states, layer_idx, cache_kwargs)


compression_chunk_size = 256

class CompressingLayer(DynamicLayer):
    def __init__(self):
        super().__init__()
        self.cumulative_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = super().update(key_states, value_states, cache_kwargs)
        B, H, T, D = keys.shape

        self.cumulative_length += key_states.shape[-2]

        layer_idx = cache_kwargs['layer_idx']

        if self.cumulative_length % compression_chunk_size == 0:
            # similarity
            c = keys @ keys.mT
            # mask diag so that self similarity of keys is not random based on length, since they're not normalized
            i = torch.arange(c.shape[-1], device=c.device)
            c[:, :, i, i] = float('nan')
            # Compute variance (manual calculation to handle NaN)
            mean = torch.nanmean(c, dim=-1, keepdim=True)
            variances = torch.nanmean((c - mean)**2, dim=-1)
            # Get top-k least similar keys (currently going to a total of cumulative_length ** 0.9 every chunk)
            num_top_k = int(self.cumulative_length ** 0.9)
            if layer_idx == 0:
                print("Compressing ", self.cumulative_length, "to", num_top_k)
            top_k_indices = torch.topk(variances, num_top_k, dim=-1, largest=False).indices
            top_k_indices = top_k_indices.view(B, H, num_top_k, 1).expand(-1, -1, -1, D)
            # use only top-k least similar key indices as retained keys and values
            keys = torch.gather(keys, 2, top_k_indices)
            values = torch.gather(values, 2, top_k_indices)
            # if layer_idx == 0:
            #     print(f"Shape of top_k_keys: {keys.shape}")
            self.keys = keys
            self.values = values

        return keys, values

# for some reason replacing transformers.cache_utils.DynamicCache with CompressingCache didn't 'take' on gsm8k although it did for mmlu, so instead we patch GenerationMixin to replace the past_key_values
import transformers.generation
class GenerationMixinReplacement(transformers.generation.GenerationMixin):
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # set up batched prefill
        self.generation_config.prefill_chunk_size = compression_chunk_size
        
        if not isinstance(past_key_values, CompressingCache):
            past_key_values = CompressingCache()
        return super().prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds, cache_position, **kwargs)

transformers.generation.GenerationMixin = GenerationMixinReplacement

# Replace DynamicCache with our custom implementation (this works for non-generate evals)
import transformers.cache_utils
transformers.cache_utils.DynamicCache = CompressingCache

from lm_eval.__main__ import cli_evaluate # if we want to use lm-eval-harness
# from eval.eval import cli_evaluate # if we want to use evalchemy
if __name__ == '__main__':
    cli_evaluate()
