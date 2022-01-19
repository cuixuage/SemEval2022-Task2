import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import math
import json
import os

class CrossAttention(nn.Module):
    """
        class BertSelfAttention(nn.Module):
            https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_bert.html
            计算代码均是self-attention一致的
        区别:
            1. 初始函数不再通过dict_config赋值
    """
    def __init__(self, num_attention_heads=12, word_embedding_dimension=768, attention_dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.word_embedding_dimension = word_embedding_dimension
        self.attention_dropout = attention_dropout
        self.attention_head_size = int(768 / 12)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(768, self.all_head_size)
        self.key = nn.Linear(768, self.all_head_size)
        self.value = nn.Linear(768, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        features: Dict[str, Tensor],
        hidden_states=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        """
            # 2021.12.27 正常逻辑decoder: encoder_attention_mask是KV的Padding Mask
        """
        ############################################################### 2021.12.30 新加入以下代码
        hidden_states = features['token_embeddings']
        ###############################################################
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        # return outputs
        ############################################################### 2021.12.30 新加入以下代码
        features.update({'cross_token_embeddings': outputs[0]})
        return features
        ###############################################################

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'cross_attention_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))
    
    def get_config_dict(self):
        return {'num_attention_heads': self.num_attention_heads,
                'word_embedding_dimension': self.word_embedding_dimension,
                'attention_dropout': self.attention_dropout}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'cross_attention_config.json'), 'r') as fIn:
            config = json.load(fIn)

        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'))
        model = CrossAttention(**config)
        model.load_state_dict(weights)
        return model