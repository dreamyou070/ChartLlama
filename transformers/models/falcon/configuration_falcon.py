# coding=utf-8
# Copyright 2023 the Falcon authors and HuggingFace Inc. team.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Falcon configuration"""
from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tiiuae/falcon-40b": "https://huggingface.co/tiiuae/falcon-40b/resolve/main/config.json",
    "tiiuae/falcon-7b": "https://huggingface.co/tiiuae/falcon-7b/resolve/main/config.json",
}


class FalconConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FalconModel`]. It is used to instantiate a Falcon
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the
    [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 65024):
            Vocabulary size of the Falcon model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`FalconModel`]
        hidden_size (`int`, *optional*, defaults to 4544):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 71):
            Number of attention heads for each attention layer in the Transformer encoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models). Only relevant if
            `config.is_decoder=True`.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for MLP layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for attention layers.
        num_kv_heads (`int`, *optional*):
            Number of key-value heads to use per attention layer. If unset, defaults to the same value as
            `num_attention_heads`.
        alibi (`bool`, *optional*, defaults to `False`):
            Whether to use ALiBi positional biases during self-attention.
        new_decoder_architecture (`bool`, *optional*, defaults to `False`):
            Whether to use the new (Falcon-40B) decoder architecture. If `True`, the `multi_query` and `parallel_attn`
            arguments are ignored, as the new decoder always uses parallel attention.
        multi_query (`bool`, *optional*, defaults to `True`):
            Whether to use multi-query attention in the decoder. Ignored when `new_decoder_architecture` is `True`.
        parallel_attn (`bool`, *optional*, defaults to `True`):
            Whether to compute attention in parallel with the feedforward layer. If False, they are consecutive
            instead, as in the original Transformer architecture. Ignored when `new_decoder_architecture` is `True`.
        bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias on Linear layers.
        bos_token_id (`int`, *optional*, defaults to 11):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 11):
            The id of the "end-of-sequence" token.

    Example:

    ```pytho
    >>> from transformers_sy import FalconModel, FalconConfig

    >>> # Initializing a small (2-layer) Falcon configuration
    >>> configuration = FalconConfig(num_hidden_layers=2)

    >>> # Initializing a model from the small configuration
    >>> model = FalconModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "falcon"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=65024,
        hidden_size=4544,
        num_hidden_layers=32,
        num_attention_heads=71,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_kv_heads=None,
        alibi=False,
        new_decoder_architecture=False,
        multi_query=True,
        parallel_attn=True,
        bias=False,
        bos_token_id=11,
        eos_token_id=11,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_kv_heads = num_attention_heads if num_kv_heads is None else num_kv_heads
        self.alibi = alibi
        self.new_decoder_architecture = new_decoder_architecture
        self.multi_query = multi_query  # Ignored when new_decoder_architecture is True
        self.parallel_attn = parallel_attn
        self.bias = bias

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary(self):
        return not self.alibi
