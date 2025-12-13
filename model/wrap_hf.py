import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Any
from transformers import Cache
from contextlib import contextmanager

def radlads_replacement_attn(module, *args, **kwargs):
    # forward while removing underscores we added to certain kwargs
    return module.attn_replacement(*args, **{k.rstrip('_'):v for k,v in kwargs.items()}), None

from transformers.modeling_utils import AttentionInterface
AttentionInterface.register('radlads_replacement_attn', radlads_replacement_attn)

@contextmanager
def replace_class(module, name, replacement):
    # purposely not using try/finally
    tmp = getattr(module, name)
    setattr(module, name, replacement)
    yield
    setattr(module, name, tmp)

def class_name_and_module_from_path(class_path):
    import importlib
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    target_class = getattr(module, class_name)
    return target_class, class_name, module

def create_model_class(replacement_attention_class, base_model_path, base_attention_path, replacement_cache_layer_class=None):
    base_model_class, _, _ = class_name_and_module_from_path(base_model_path)
    base_model_attention_class, base_model_attention_class_name, base_model_attention_module = class_name_and_module_from_path(base_attention_path)

    # subclass of original attention class that owns an instance of our replacement and calls it for attention
    class RADLADSAttentionWrapper(base_model_attention_class):
        def __init__(self, config, layer_idx):
            #print("RADLADSAttentionWrapper.__init__")
            super().__init__(config, layer_idx)
            self.teacher_attn = None
            if not self.is_original_attention_layer():
                self.attn_replacement = replacement_attention_class(config, layer_idx)

                # in stage 1 distillation, we also have a separate teacher attention on layers that are changed
                if getattr(config, 'radlads_distillation_stage', 0) == 1:
                    self.teacher_attn = base_model_attention_class()

        def is_original_attention_layer(self):
            return self.config.layer_hybrid_types[self.layer_idx] == 'full_attention'

        def forward(self, use_cache:bool, past_key_values:Optional[Cache], **kwargs):
            #print("RADLADSAttentionWrapper.forward")

            post_attention_hidden_states = getattr(kwargs, 'post_attention_hidden_states', [])
            # add second underscored copies so things like hidden_states get passed through to attention function via kwargs
            kwargs = kwargs | dict(
                hidden_states_=kwargs['hidden_states'],
                position_embeddings_=kwargs['position_embeddings'],
                use_cache_=use_cache,
                past_key_values_=past_key_values,
                cache_position_=kwargs['cache_position'],
            )

            # update affected cache layers to use our replacement cache layer class
            if use_cache and past_key_values is not None and replacement_cache_layer_class is not None:
                while len(past_key_values.layers) <= self.layer_idx:
                    past_key_values.layers.append(past_key_values.layer_class_to_replicate())
                if not isinstance(past_key_values.layers[self.layer_idx], replacement_cache_layer_class):
                    past_key_values.layers[self.layer_idx] = replacement_cache_layer_class()

            # temporarily change config
            if not self.is_original_attention_layer():
                original_attn_implementation = self.config._attn_implementation
                self.config._attn_implementation = 'radlads_replacement_attn'

            student_output = super().forward(use_cache=use_cache, past_key_values=past_key_values, **kwargs)

            # then revert the config
            if not self.is_original_attention_layer():
                self.config._attn_implementation = original_attn_implementation

            if self.teacher_attn is not None:
                # in stage 1 we need to return the post attention hidden states for student and teacher, but use teacher as the actual model output for this layer
                teacher_output = self.teacher_attn.forward(**kwargs)
                layer_output = teacher_output
            else:
                teacher_output = None
                layer_output = student_output
            post_attention_hidden_states += [(student_output, teacher_output)]

            return layer_output

    # FIXME - would have to temporarily change GenerationMixin superclass while creating this I guess using replace_class
    # as for changing the cache created at runtime in the model itself I'm not sure how to do that without editing the model code or globally permanently swapping out DynamicCache
    class RADLADSForCausalLM(base_model_class):
        def __init__(self, config):
            #print("RADRWKV7cHybridForCausalLM.__init__")
            # replace attention classes during construction
            with replace_class(
                base_model_attention_module, 
                base_model_attention_class_name, 
                RADLADSAttentionWrapper
            ):
                super().__init__(config)

    return RADLADSForCausalLM

from transformers.cache_utils import CacheLayerMixin
class StaticStateCacheLayer(CacheLayerMixin):
    def lazy_initialization(self):
        self.keys = None
        self.values = None
        self.is_initialized = True    

    def update(
        self,
        key_states, 
        value_states,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ):
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization()

        old_keys, old_values = self.keys, self.values
        self.keys = key_states
        self.values = value_states
        return old_keys, old_values

def create_config_class(base_configuration_path):
    base_config_class, _, _ = class_name_and_module_from_path(base_configuration_path)

    class RADLADSConfig(base_config_class):
        def __init__(self, layer_hybrid_types: Optional[list[str]] = None, **kwargs):
            super().__init__(**kwargs)
            if layer_hybrid_types is None:
                layer_hybrid_types = ['replacement_attention'] * self.num_hidden_layers
            self.layer_hybrid_types = layer_hybrid_types

    return RADLADSConfig
