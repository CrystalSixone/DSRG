import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel

from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad


logger = logging.getLogger(__name__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    BertLayerNorm = torch.nn.LayerNorm

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class AggregateAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AggregateAttention, self).__init__()
        self.attn = nn.Parameter(torch.Tensor(hidden_size,1))
        nn.init.uniform_(self.attn, -0.1, 0.1)
    
    def forward(self,x):
        M = torch.tanh(x)
        a = F.softmax(torch.matmul(M,self.attn),1) # [bs,max_len,1]
        out = torch.sum(x * a,1) # [bs,768]
        x = torch.tanh(out)
        return x
    
    def forward_mean(self,x,x_mask):
        nums = torch.sum(x_mask.squeeze(-1), 1, keepdim=True)
        nums_change = torch.where(nums==0,torch.full_like(nums,1),nums)
        x_mean = torch.sum(x * x_mask, 1) / nums_change
        return x_mean

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # For action and object
        self.act_embedding = nn.Embedding(config.word_max_action,config.hidden_size) 
        self.obj_embedding = nn.Embedding(config.word_max_object,config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, act_txt=None, obj_txt=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        act_embeddings = self.act_embedding(act_txt)
        obj_embeddings = self.obj_embedding(obj_txt)

        return embeddings, act_embeddings, obj_embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.config = config
        self.extra_dropout = nn.Dropout(0.2)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        hidden_states: (N, L_{hidden}, D)
        attention_mask: (N, H, L_{hidden}, L_{hidden})
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
    

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
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

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)


    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask,
                None if head_mask is None else head_mask[i],
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, location=0):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, location]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores

class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores
    
class BertXLayer(nn.Module):
    def __init__(self, config):
        super(BertXLayer, self).__init__()
        self.attention = BertXAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        attention_output, attention_scores = self.attention(input_tensor, ctx_tensor, ctx_att_mask)
        attention_outputs = (attention_output, attention_scores)
        intermediate_output = self.intermediate(attention_outputs[0])
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

class BertXEncoder(nn.Module):
    def __init__(self, config):
        super(BertXEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertXLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs, ctx, attention_mask):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(inputs, ctx, attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class GraphLXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Lang self-att and FFN layer
        if config.use_lang2visn_attn: # default False
            self.lang_self_att = BertAttention(config)
            self.lang_inter = BertIntermediate(config)
            self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
        graph_sprels=None
    ):  
        visn_att_output = self.visual_attention(
            visn_feats, lang_feats, ctx_att_mask=lang_attention_mask
        )[0]

        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels
        visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]

        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)

        return visn_output

    def forward_lang2visn(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
    ):
        lang_att_output = self.visual_attention(
            lang_feats, visn_feats, ctx_att_mask=visn_attention_mask
        )[0]
        lang_att_output = self.lang_self_att(
            lang_att_output, lang_attention_mask
        )[0]
        lang_inter_output = self.lang_inter(lang_att_output)
        lang_output = self.lang_output(lang_inter_output, lang_att_output)
        return lang_output

class LanguageEncoder(nn.Module):
    # add action embedding and object embedding
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False
        
        # for action and object
        self.act_obj_token_emb = nn.Embedding(2,config.hidden_size) # 0 for act, 1 for obj
        self.instr_actobj_encoder = BertOutAttention(config)
        self.instr_aug_linear = nn.Linear(config.hidden_size,1)
        self.instr_ori_linear = nn.Linear(config.hidden_size,1)
        self.instr_sigmoid = nn.Sigmoid()

        self.instr_fuse_layernorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, txt_embeds, txt_masks, act_embeds=None, act_masks=None, obj_embeds=None, obj_masks=None):
        '''
        :param act_masks: [batch_size, act_txt_len]
        '''
        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]

        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        
        if act_embeds is not None:
            act_obj_embeds = torch.cat((act_embeds, obj_embeds),1) 
            extended_aug_masks = extend_neg_masks(torch.cat((act_masks,obj_masks),1)) 
            instr_aug_embeds = self.instr_actobj_encoder(txt_embeds, act_obj_embeds, extended_aug_masks)[0]
            aug_linear_weight = self.instr_aug_linear(instr_aug_embeds)
            ori_linear_weight = self.instr_ori_linear(txt_embeds)
            aug_weight = self.instr_sigmoid(aug_linear_weight+ori_linear_weight)
            txt_embeds = torch.mul(aug_weight,instr_aug_embeds) + torch.mul((1-aug_weight),txt_embeds)  
            
        return txt_embeds
    
class CrossmodalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_x_layers = config.num_x_layers
        self.x_layers = nn.ModuleList(
            [GraphLXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):
        extended_txt_masks = extend_neg_masks(txt_masks)
        extended_img_masks = extend_neg_masks(img_masks) # (N, 1(H), 1(L_q), L_v)
        for i, layer_module in enumerate(self.x_layers):
            img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                img_embeds, extended_img_masks,
                graph_sprels=graph_sprels
            )
        return img_embeds

class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size + 3, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        
        self.adaptive_pano_attn = nn.Linear(config.hidden_size,1) 
        self.adaptive_pano_act = ACT2FN[config.hidden_act]
        self.adaptive_softmax = nn.Softmax(dim=1)

        self.obj_bbox_linear = nn.Linear(config.angle_feat_size + 3,config.obj_v2_size)
        self.obj_name_linear = nn.Embedding(config.obj_name_vocab_size, config.obj_v2_size)
        self.obj_nav_type_embedding = nn.Embedding(3, config.obj_v2_size)
        self.obj_layer_norm = BertLayerNorm(config.obj_v2_size, eps=1e-12)

        obj_self_config = copy.copy(config)
        obj_self_config.num_hidden_layers = 2
        obj_self_config.hidden_size = config.obj_v2_size
        obj_self_config.num_attention_heads = config.obj_v2_size // 64 # 64*2=128
        self.obj_self_encoder = BertEncoder(obj_self_config)
            
        self.obj_attn = AggregateAttention(config.obj_v2_size)
        self.vis_obj_linear = nn.Sequential(
            nn.Linear(config.hidden_size + config.obj_v2_size, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )

        # 0: non-navigable, 1: navigable, 2: object
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pano_encoder = create_transformer_encoder(
            config, config.num_pano_layers, norm=True
        )
        
class LocalVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.encoder = CrossmodalEncoder(config)

class GlobalMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)
        
        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None
    
class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None, output_size=1):
        super().__init__()
        if input_size is None:
            input_size = hidden_size

        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                nn.ReLU(),
                                BertLayerNorm(hidden_size, eps=1e-12),
                                nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.net(x)

class GlocalTextPathNavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.lang_encoder = LanguageEncoder(config)

        self.img_embeddings = ImageEmbeddings(config)
        
        self.local_encoder = LocalVPEncoder(config)
        self.global_encoder = GlobalMapEncoder(config)

        self.global_sap_head = ClsPrediction(self.config.hidden_size)
        self.local_sap_head = ClsPrediction(self.config.hidden_size)
        if config.glocal_fuse:
            self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
        else:
            self.sap_fuse_linear = None
        if self.config.obj_feat_size > 0:
            self.og_head = ClsPrediction(self.config.hidden_size)
        
        self.extra_drop = nn.Dropout(0.2)

        # For recurrent memory fusion 
        self.gmap_pooler = BertPooler(config)
        self.vp_pooler = BertPooler(config)
        self.txt_pooler = BertPooler(config)
        self.local_his_map = nn.Linear(config.hidden_size*3, config.hidden_size)
        self.local_his_ln = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.drop_env = nn.Dropout(p=config.feat_dropout)

        self.init_weights()
        
        if config.fix_lang_embedding or config.fix_local_branch:
            for k, v in self.embeddings.named_parameters():
                v.requires_grad = False
            for k, v in self.lang_encoder.named_parameters():
                v.requires_grad = False
        if config.fix_pano_embedding or config.fix_local_branch:
            for k, v in self.img_embeddings.named_parameters():
                v.requires_grad = False
        if config.fix_local_branch:
            for k, v in self.local_encoder.named_parameters():
                v.requires_grad = False
            for k, v in self.local_sap_head.named_parameters():
                v.requires_grad = False
            for k, v in self.og_head.named_parameters():
                v.requires_grad = False
    
    def forward_text(self, txt_ids, txt_masks, act_txt=None, act_txt_masks=None, obj_txt=None, obj_txt_masks=None):
        txt_token_type_ids = torch.zeros_like(txt_ids)
        if obj_txt is not None:
            txt_embeds, act_embeds, obj_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids,
                act_txt=act_txt, obj_txt=obj_txt
            )
            txt_embeds = self.lang_encoder(txt_embeds, txt_masks, act_embeds, act_txt_masks, obj_embeds, obj_txt_masks)
        else:
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
            txt_embeds = self.lang_encoder(txt_embeds, txt_masks)
        return txt_embeds

    def forward_panorama_per_step(self,view_img_fts, obj_img_fts, loc_fts, nav_types, view_lens, obj_lens):
        device = view_img_fts.device

        view_img_embeds = self.img_embeddings.img_layer_norm(
            self.img_embeddings.img_linear(view_img_fts)
        )

        bs, view_len, obj_len = obj_img_fts.shape[0], obj_img_fts.shape[1], obj_img_fts.shape[2]
        obj_bbox_fts = obj_img_fts[:,:,:,:-1]
        obj_name_fts = obj_img_fts[:,:,:,-1].int()
        obj_bbox_fts_linear = self.img_embeddings.obj_bbox_linear(obj_bbox_fts)
        obj_name_fts_linear = self.img_embeddings.obj_name_linear(obj_name_fts)
        obj_fuse_linear = obj_bbox_fts_linear + obj_name_fts_linear

        # add nav type to obj
        obj_nav_fts = self.img_embeddings.obj_nav_type_embedding(nav_types).unsqueeze(-2) 
        obj_fuse_linear = obj_nav_fts + obj_fuse_linear

        obj_fts = self.img_embeddings.obj_layer_norm(obj_fuse_linear)

        obj_fts_re = rearrange(obj_fts,"b v o d -> b (v o) d")
        obj_re_masks = gen_seq_masks(obj_lens.reshape(-1)).reshape(-1,1,1,view_len*obj_len) # [bs*v_len*obj_len]
        obj_fts = self.img_embeddings.obj_self_encoder(obj_fts_re,obj_re_masks)[0]
        obj_fts = rearrange(obj_fts,"b (v o) d -> b v o d",v=view_len)
        traj_obj_embeds = obj_fts.mean(dim=2) # [bs,v,d]
            
        traj_img_embeds = view_img_embeds
        traj_vp_lens = view_lens
        
        pano_embeds = traj_img_embeds + \
                    self.img_embeddings.loc_layer_norm(self.img_embeddings.loc_linear(loc_fts)) + \
                    self.img_embeddings.nav_type_embedding(nav_types) + \
                    self.embeddings.token_type_embeddings(torch.ones(1, 1).long().to(device))
        pano_embeds = self.img_embeddings.layer_norm(pano_embeds)
        pano_embeds = self.img_embeddings.dropout(pano_embeds)
        pano_masks = gen_seq_masks(traj_vp_lens)
        traj_self_embeds = self.img_embeddings.pano_encoder(
            pano_embeds, src_key_padding_mask=pano_masks.logical_not()
        )

        # image & obj fusion
        vis_obj_embeds = torch.cat((traj_self_embeds,traj_obj_embeds),-1) # [bs,view_len,dim+obj_dim]
        traj_embeds = self.img_embeddings.vis_obj_linear(vis_obj_embeds) # [bs,view_len,dim]
        
        # For adaptively aggregation
        traj_ori_embeds = traj_embeds.clone()
        traj_fused_weight = self.img_embeddings.adaptive_pano_attn(traj_ori_embeds)
        traj_fused_weight_act = torch.tanh(traj_fused_weight) 
        traj_fused_weight_act = self.img_embeddings.adaptive_softmax(traj_fused_weight_act)
        traj_fused_embeded_update = torch.mul(traj_ori_embeds,traj_fused_weight_act)
        traj_fused_embeds = torch.sum(traj_fused_embeded_update,dim=1)

        return traj_embeds, pano_masks, traj_fused_embeds


    def forward_navigation_per_step(
        self, txt_embeds, txt_masks, gmap_img_embeds, gmap_step_ids, gmap_pos_fts, 
        gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids,
        vp_img_embeds, vp_pos_fts, vp_masks, vp_nav_masks, vp_obj_masks, vp_cand_vpids
    ):  
        batch_size = txt_embeds.size(0)

        # global branch
        gmap_embeds = gmap_img_embeds + \
                    self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                    self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)

        if self.global_encoder.sprel_linear is not None:
            graph_sprels = self.global_encoder.sprel_linear(
                gmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.global_encoder.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
       
        # local branch
        vp_embeds = vp_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)
        vp_embeds = self.local_encoder.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)

        # navigation logits
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else: # default
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)

        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))
        local_logits.masked_fill_(vp_nav_masks.logical_not(), -float('inf'))
        
        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]): # Process Local branch
                if j > 1: # jump over the [stop] and [MEM] token
                    if cand_vpid in visited_nodes:
                        bw_logits += local_logits[i, j]
                    else:
                        tmp[cand_vpid] = local_logits[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 1 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits
        
        # object grounding logits
        if vp_obj_masks is not None:
            obj_logits = self.og_head(vp_embeds).squeeze(2)
            obj_logits.masked_fill_(vp_obj_masks.logical_not(), -float('inf'))
        else:
            obj_logits = None 
        
        # represent the reasoning information for each step.
        gmap_cls_embed = self.gmap_pooler(gmap_embeds,location=0)
        vp_cls_embeds = self.vp_pooler(vp_embeds,location=0)
        txt_cls_embeds = self.txt_pooler(txt_embeds,location=0)
        cls_embeds = self.local_his_ln(self.local_his_map(torch.cat((gmap_cls_embed,vp_cls_embeds,txt_cls_embeds),dim=-1)))

        outs = {
            'gmap_embeds': gmap_embeds,
            'vp_embeds': vp_embeds,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'fused_logits': fused_logits,
            'obj_logits': obj_logits,
            'txt_embeds': txt_embeds,
            'cls_embeds': cls_embeds
        }
        return outs
    
    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked
    
    def forward(self, mode, batch, **kwargs):
        if mode == 'language':
            txt_embeds = self.forward_text(batch['txt_ids'], batch['txt_masks'], 
                batch['act_txt_ids'], batch['act_txt_masks'], batch['obj_txt_ids'], batch['obj_txt_masks'])
            return txt_embeds

        elif mode == 'panorama':
            pano_embeds, pano_masks, pano_fused_embeds = self.forward_panorama_per_step(
                batch['view_img_fts'], batch['obj_img_fts'], batch['loc_fts'],
                batch['nav_types'], batch['view_lens'], batch['obj_lens']
            )
            return pano_embeds, pano_masks, pano_fused_embeds
            
        elif mode == 'navigation':
             return self.forward_navigation_per_step(
                batch['txt_embeds'], batch['txt_masks'], batch['gmap_img_embeds'], 
                batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_masks'],
                batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'], 
                batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'],
                batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids']
            )