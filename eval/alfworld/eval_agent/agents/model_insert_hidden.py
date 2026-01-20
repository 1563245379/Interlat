import json
from typing import Dict, Optional, Sequence, Tuple, Union, List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import transformers

import random
import time
from tqdm import tqdm
import math


# class ModelWithInsertedHiddenState(nn.Module):
#     """Wrapper model that inserts hidden states at specific positions in conversations."""

#     def __init__(self, base_model, prepended_length, hidden_size, prepended_learnable=False, num_heads=8,
#                 plan_similarity_weight=0.5, random_contrast_weight=1.5):
#         super().__init__()
#         self.base_model = base_model
#         self.prepended_length = prepended_length
#         self.prepended_learnable = prepended_learnable
#         self.config = base_model.config
#         self.tokenizer = None

#         # 新增正则化权重参数
#         self.hidden_state_weight = nn.Parameter(torch.tensor(1.0))
#         # 调整权重参数
#         self.hidden_state_weight = nn.Parameter(torch.tensor(1.0))
#         self.contrastive_weight = 0.1
#         self.plan_similarity_weight = plan_similarity_weight
#         self.random_contrast_weight = random_contrast_weight
        
#         # 新增：范围控制参数
#         self.loss_range_mode = "extended"  # "local", "extended", "global", "adaptive"
#         self.context_window = 32  # 扩展窗口大小

#         # 如果需要learnable的默认hidden state（作为fallback）
#         if prepended_learnable:
#             self.default_prepended_hidden_state = nn.Parameter(
#                 torch.randn(prepended_length, hidden_size) * 0.02
#             )
#         else:
#             self.register_buffer(
#                 'default_prepended_hidden_state',
#                 torch.zeros(prepended_length, hidden_size)
#             )

#     def process_hidden_states(self, hidden_states):
#         """Process hidden states through MHA layer"""
#         # normed_hidden = self.pre_ln(hidden_states)
#         # attn_output, _ = self.hidden_mha(
#         #     query=normed_hidden,
#         #     key=normed_hidden,
#         #     value=normed_hidden
#         # )
#         # hidden_states = hidden_states + attn_output
#         # hidden_states = self.post_ln(hidden_states)
#         return hidden_states

#     # def adjust_weights_dynamically(self, plan_similarity_loss, random_contrast_loss):
#     #     """基于loss动态调整权重"""
        
#     #     # random_contrast_loss: 0->0.1, 0.5->3.0
#     #     contrast_normalized = torch.clamp(random_contrast_loss / 0.5, 0.0, 1.0)
#     #     self.random_contrast_weight = 0.1 + contrast_normalized * 2.9  # 0.1 + [0,1] * 2.9 = [0.1, 3.0]
        
#     #     # plan_similarity_loss: 0->0.1, 1.0->2.0 (设1.0为上限，给超过0.8的情况留余量)
#     #     plan_normalized = torch.clamp(plan_similarity_loss / 1.0, 0.0, 1.0)
#     #     self.plan_similarity_weight = 0.1 + plan_normalized * 0.9  # 0.1 + [0,1] * 1.9 = [0.1, 1.0]

#     def adjust_weights_dynamically(self, plan_similarity_loss, random_contrast_loss):
#         """基于loss动态调整权重"""
        
#         # Option 1: Use Python's built-in min/max functions for scalar values
#         # random_contrast_loss: 0->0.1, 0.5->3.0
#         contrast_normalized = max(0.0, min(1.0, random_contrast_loss / 0.5))
#         self.random_contrast_weight = 0.1 + contrast_normalized * 2.9  # 0.1 + [0,1] * 2.9 = [0.1, 3.0]
        
#         # plan_similarity_loss: 0->0.1, 1.0->2.0 (设1.0为上限，给超过0.8的情况留余量)
#         plan_normalized = max(0.0, min(1.0, plan_similarity_loss / 1.0))
#         self.plan_similarity_weight = 0.1 + plan_normalized * 0.9  # 0.1 + [0,1] * 1.9 = [0.1, 1.0]

#     def insert_plan_tokens(self, input_ids, attention_mask, labels, human_end_positions, plans):
#         """将plan文本插入到指定位置，前后加入特殊标记"""
#         batch_size = input_ids.shape[0]
#         new_input_ids = []
#         new_attention_mask = []
#         new_labels = []

#         # 获取特殊标记的token ids
#         bop_token_id = self.tokenizer.convert_tokens_to_ids('<bop>')
#         eop_token_id = self.tokenizer.convert_tokens_to_ids('<eop>')

#         for batch_idx in range(batch_size):
#             if human_end_positions[batch_idx] >= 0 and plans[batch_idx]:
#                 insert_pos = human_end_positions[batch_idx].item()

#                 # 获取plan的token ids
#                 plan_token_ids = torch.tensor(plans[batch_idx], device=input_ids.device)

#                 # 分割原始序列
#                 before_ids = input_ids[batch_idx, :insert_pos]
#                 after_ids = input_ids[batch_idx, insert_pos:]

#                 # 创建带有特殊标记的plan序列: <bop> + plan + <eop>
#                 bop_tensor = torch.tensor([bop_token_id], device=input_ids.device)
#                 eop_tensor = torch.tensor([eop_token_id], device=input_ids.device)
#                 marked_plan_ids = torch.cat([bop_tensor, plan_token_ids, eop_tensor], dim=0)

#                 # 插入带标记的plan tokens
#                 new_seq = torch.cat([before_ids, marked_plan_ids, after_ids], dim=0)

#                 new_input_ids.append(new_seq)

#                 # 处理attention mask
#                 if attention_mask is not None:
#                     before_mask = attention_mask[batch_idx, :insert_pos]
#                     after_mask = attention_mask[batch_idx, insert_pos:]
#                     plan_mask = torch.ones(len(marked_plan_ids), device=attention_mask.device)

#                     new_mask = torch.cat([before_mask, plan_mask, after_mask], dim=0)

#                     new_attention_mask.append(new_mask)

#                 # 处理labels
#                 if labels is not None:
#                     before_labels = labels[batch_idx, :insert_pos]
#                     after_labels = labels[batch_idx, insert_pos:]
#                     plan_labels = torch.full((len(marked_plan_ids),), IGNORE_TOKEN_ID, device=labels.device)

#                     new_label = torch.cat([before_labels, plan_labels, after_labels], dim=0)

#                     new_labels.append(new_label)
#             else:
#                 # 没有插入位置或plan，保持原样
#                 new_input_ids.append(input_ids[batch_idx])
#                 if attention_mask is not None:
#                     new_attention_mask.append(attention_mask[batch_idx])
#                 if labels is not None:
#                     new_labels.append(labels[batch_idx])

#         result = {
#             'input_ids': pad_sequence(new_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
#             'attention_mask': pad_sequence(new_attention_mask, batch_first=True, padding_value=0) if new_attention_mask else None,
#             'labels': pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID) if new_labels else None
#         }

#         return result

#     # def generate_random_hidden_states(self, batch_size, device, dtype, hidden_state_seq_len):
#     #     """生成随机的hidden states作为对比"""
#     #     # return torch.randn(
#     #     #     batch_size, self.prepended_length, self.config.hidden_size,
#     #     #     device=device, dtype=dtype
#     #     # ) * 0.02
#     #     assert len(hidden_state_seq_len) == batch_size, \
#     #         "hidden_state_seq_len 的长度必须等于 batch_size"

#     #     hidden_states = []
#     #     for i in range(batch_size):
#     #         seq_len_i = hidden_state_seq_len[i]
#     #         hidden_state_i = torch.randn(
#     #             seq_len_i, self.config.hidden_size,
#     #             device=device, dtype=dtype
#     #         ) * 0.02  # 缩放因子保持不变
#     #         hidden_states.append(hidden_state_i)

#     #     return hidden_states

#     def generate_random_hidden_states(self, batch_size, device, dtype, hidden_state_seq_len):
#         """生成更合理的随机hidden states"""
#         hidden_states = []
        
#         for i in range(batch_size):
#             seq_len_i = hidden_state_seq_len[i]
            
#             # 方案1: 从同一batch中随机选择其他样本的hidden state
#             if len(hidden_state_seq_len) > 1:
#                 # 随机选择不同的索引
#                 other_indices = [j for j in range(batch_size) if j != i]
#                 if other_indices:
#                     random_idx = random.choice(other_indices)
#                     # 使用其他样本的hidden state作为"随机"对比
#                     # 这样更有挑战性，因为都是真实的hidden state
#                     if hasattr(self, '_current_batch_hidden_states') and self._current_batch_hidden_states:
#                         other_hidden = self._current_batch_hidden_states[random_idx]
#                         # 调整长度匹配
#                         if other_hidden.shape[0] >= seq_len_i:
#                             hidden_state_i = other_hidden[:seq_len_i]
#                         else:
#                             # 如果长度不够，用其他样本的重复+噪声
#                             repeat_times = (seq_len_i // other_hidden.shape[0]) + 1
#                             repeated = other_hidden.repeat(repeat_times, 1)[:seq_len_i]
#                             # 添加小噪声
#                             noise = torch.randn_like(repeated) * 0.01
#                             hidden_state_i = repeated + noise
#                         hidden_states.append(hidden_state_i)
#                         continue
#             else:
#                 # 方案2: 改进的随机生成 - 模拟真实hidden state的分布
#                 # 使用更接近真实hidden state的统计特性
#                 hidden_state_i = torch.randn(
#                     seq_len_i, self.config.hidden_size,
#                     device=device, dtype=dtype
#                 )
                
#                 # 添加类似真实hidden state的结构
#                 # 1. 降低方差，真实hidden state通常不会有太大的值
#                 hidden_state_i = hidden_state_i * 0.1
                
#                 # 2. 添加一些周期性模式（模拟attention pattern）
#                 if seq_len_i > 4:
#                     for j in range(0, seq_len_i, 4):
#                         end_j = min(j + 4, seq_len_i)
#                         # 在小窗口内添加相关性
#                         if j > 0:
#                             hidden_state_i[j:end_j] = 0.7 * hidden_state_i[j:end_j] + 0.3 * hidden_state_i[j-1:j-1+end_j-j]
                
#                 # 3. 归一化到合理范围
#                 hidden_state_i = F.normalize(hidden_state_i, dim=-1) * 2.0
                
#                 hidden_states.append(hidden_state_i)
        
#         return hidden_states

#     def _forward_with_hidden_states(
#             self, input_ids, attention_mask, inputs_embeds, labels,
#             human_end_positions, prepended_hidden_states,  # 注意：这是 list of tensor
#             past_key_values, use_cache, output_attentions,
#             output_hidden_states, return_dict, **kwargs
#     ):
#         device = next(self.base_model.parameters()).device
#         model_dtype = next(self.base_model.parameters()).dtype

#         if input_ids is not None and inputs_embeds is None:
#             inputs_embeds = self.base_model.get_input_embeddings()(input_ids).to(model_dtype)

#         inputs_embeds = inputs_embeds.to(dtype=model_dtype)

#         # 如果提供了 prepended_hidden_states，则处理每个 batch 对应的 hidden state
#         if prepended_hidden_states is not None:
#             prepended_hidden_states = [h.to(dtype=model_dtype) for h in prepended_hidden_states]

#         # 获取特殊标记的embeddings
#         bop_token_id = self.tokenizer.convert_tokens_to_ids('<bop>')
#         eop_token_id = self.tokenizer.convert_tokens_to_ids('<eop>')
        
#         bop_embedding = self.base_model.get_input_embeddings()(torch.tensor([bop_token_id], device=device)).squeeze(0)
#         eop_embedding = self.base_model.get_input_embeddings()(torch.tensor([eop_token_id], device=device)).squeeze(0)

#         batch_size = inputs_embeds.shape[0]
#         new_inputs_embeds = []
#         new_attention_mask = [] if attention_mask is not None else None
#         new_labels = [] if labels is not None else None

#         for batch_idx in range(batch_size):
#             insert_pos = human_end_positions[batch_idx].item()
#             if insert_pos < 0:
#                 # 没有插入位置，保持原样
#                 new_inputs_embeds.append(inputs_embeds[batch_idx])
#                 if attention_mask is not None:
#                     new_attention_mask.append(attention_mask[batch_idx])
#                 if labels is not None:
#                     new_labels.append(labels[batch_idx])
#                 continue

#             # 获取原始输入嵌入
#             before = inputs_embeds[batch_idx, :insert_pos]
#             after = inputs_embeds[batch_idx, insert_pos:]

#             # 获取当前 batch 的 prepended hidden state
#             if prepended_hidden_states is not None and batch_idx < len(prepended_hidden_states):
#                 hidden_state_to_use = prepended_hidden_states[batch_idx]
#             # else:
#             #     hidden_state_to_use = self.default_prepended_hidden_state

#             # Process hidden_state_to_use through MHA
#             hidden_state_to_use = self.process_hidden_states(hidden_state_to_use.unsqueeze(0)).squeeze(0)

#             # 在hidden state前后添加特殊标记embedding: <bop> + hidden_state + <eop>
#             marked_hidden_state = torch.cat([
#                 bop_embedding.unsqueeze(0),  # <bop>
#                 hidden_state_to_use,         # hidden states
#                 eop_embedding.unsqueeze(0)   # <eop>
#             ], dim=0).to(dtype=model_dtype) 

#             # 拼接
#             batch_embeds = torch.cat([before, marked_hidden_state, after], dim=0)
#             new_inputs_embeds.append(batch_embeds)

#             # 处理 attention_mask
#             if attention_mask is not None:
#                 before_mask = attention_mask[batch_idx, :insert_pos]
#                 after_mask = attention_mask[batch_idx, insert_pos:]
#                 prepended_mask = torch.ones(
#                     marked_hidden_state.size(0),  # 包含特殊标记的长度
#                     dtype=attention_mask.dtype,
#                     device=attention_mask.device
#                 )
#                 batch_mask = torch.cat([before_mask, prepended_mask, after_mask], dim=0)
#                 new_attention_mask.append(batch_mask)

#             # 处理 labels
#             if labels is not None:
#                 before_labels = labels[batch_idx, :insert_pos]
#                 after_labels = labels[batch_idx, insert_pos:]
#                 prepended_labels = torch.full(
#                     (marked_hidden_state.size(0),),  # 包含特殊标记的长度
#                     IGNORE_TOKEN_ID,
#                     dtype=labels.dtype,
#                     device=labels.device
#                 )
#                 batch_labels = torch.cat([before_labels, prepended_labels, after_labels], dim=0)
#                 new_labels.append(batch_labels)

#         # Pad 到统一长度
#         inputs_embeds = torch.nn.utils.rnn.pad_sequence(new_inputs_embeds, batch_first=True, padding_value=0)
#         if new_attention_mask:
#             attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
#         if new_labels:
#             labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)

#         # # Debug: 检查非法值
#         # if torch.isnan(inputs_embeds).any():
#         #     raise ValueError("inputs_embeds contains NaN values!")
#         # if torch.isinf(inputs_embeds).any():
#         #     raise ValueError("inputs_embeds contains Inf values!")

#         # # Debug: 打印 inputs_embeds dtype
#         # print(f"[DEBUG] inputs_embeds.dtype = {inputs_embeds.dtype}")
#         # if attention_mask is not None:
#         #     print(f"[DEBUG] attention_mask.dtype = {attention_mask.dtype}")
#         # if labels is not None:
#         #     print(f"[DEBUG] labels.dtype = {labels.dtype}")

#         # # 检查是否为 float32
#         # if inputs_embeds.dtype == torch.float32:
#         #     print(f"[WARNING] inputs_embeds is float32! Change to {model_dtype}")
#         #     inputs_embeds = inputs_embeds.to(dtype=model_dtype)

#         return self.base_model(
#             input_ids=None,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             labels=labels,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             **kwargs
#         )

#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             past_key_values=None,
#             inputs_embeds=None,
#             labels=None,
#             plans=None,
#             use_cache=None,
#             output_attentions=None,
#             output_hidden_states=None,
#             return_dict=None,
#             human_end_positions=None,
#             prepended_hidden_states=None,
#             **kwargs
#     ):
#         device = next(self.base_model.parameters()).device
#         model_dtype = next(self.base_model.parameters()).dtype
#         self._current_batch_hidden_states = prepended_hidden_states
#         print(f'prepended_hidden_states的个数: {len(prepended_hidden_states)}')

#         # Move all inputs to the correct device
#         if input_ids is not None:
#             input_ids = input_ids.to(device)
#         if attention_mask is not None:
#             attention_mask = attention_mask.to(device)
#         if inputs_embeds is not None:
#             inputs_embeds = inputs_embeds.to(device)
#         if labels is not None:
#             labels = labels.to(device)
#         if human_end_positions is not None:
#             human_end_positions = human_end_positions.to(device)

#         # Get batch size
#         if input_ids is not None:
#             batch_size, seq_len = input_ids.shape
#         elif inputs_embeds is not None:
#             batch_size, seq_len, hidden_size = inputs_embeds.shape
#         else:
#             raise ValueError("Either input_ids or inputs_embeds must be provided")

#         # ========== 新增：生成三种情况的输出 ==========
#         hidden_state_seq_len = []
#         plan_seq_len = []
#         for item in prepended_hidden_states:
#             hidden_state_seq_len.append(item.shape[0])

#         for item in plans:
#             plan_seq_len.append(len(item))

#         # print("Input IDs:", input_ids)
#         # print("Attention mask:", attention_mask)
#         # print("Plans:", plans)
#         # print("human_end_positions:", human_end_positions)
#         # print("prepended_hidden_states:", prepended_hidden_states)

#         # 1. 正常的hidden state输出（带特殊标记）
#         normal_outputs = self._forward_with_hidden_states(
#             input_ids, attention_mask, inputs_embeds, labels,
#             human_end_positions, prepended_hidden_states,
#             past_key_values, use_cache, output_attentions,
#             output_hidden_states, return_dict, **kwargs
#         )

#         # 2. plan文本插入的输出（带特殊标记）
#         plan_outputs = None
#         if plans is not None:
#             plan_data = self.insert_plan_tokens(
#                 input_ids, attention_mask, labels, human_end_positions, plans
#             )
#             plan_outputs = self.base_model(
#                 input_ids=plan_data['input_ids'],
#                 attention_mask=plan_data['attention_mask'],
#                 labels=plan_data['labels'],
#                 past_key_values=past_key_values,
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#                 **kwargs
#             )

#         # 3. 随机hidden state的输出（带特殊标记）
#         random_hidden_states = self.generate_random_hidden_states(
#             batch_size, device, model_dtype, hidden_state_seq_len
#         )

#         random_outputs = self._forward_with_hidden_states(
#             input_ids, attention_mask, inputs_embeds, labels,
#             human_end_positions, random_hidden_states,
#             past_key_values, use_cache, output_attentions,
#             output_hidden_states, return_dict, **kwargs
#         )

#         # ========== 计算KL散度损失 ==========
#         if plan_outputs is not None and human_end_positions is not None:
#             # 计算plan相似性损失（正常hidden state应该与plan相似）
#             plan_similarity_loss = self.calculate_plan_similarity_loss(
#                 normal_outputs.logits, plan_outputs.logits, human_end_positions, hidden_state_seq_len, plan_seq_len
#             )
#             # 计算随机对比损失（正常hidden state应该与随机hidden state不同）
#             random_contrast_loss = self.calculate_random_contrast_loss(
#                 normal_outputs.logits, random_outputs.logits, human_end_positions, hidden_state_seq_len
#             )

#             # 在计算完各个loss之后添加：
#             # self.adjust_weights_dynamically(plan_similarity_loss.item(), random_contrast_loss.item())

#             # 然后正常计算总loss
#             total_loss = normal_outputs.loss + \
#                          self.plan_similarity_weight * plan_similarity_loss + \
#                          self.random_contrast_weight * random_contrast_loss

#             # 打印损失信息用于调试
#             if hasattr(self, 'training') and self.training:
#                 print(f"Normal loss: {normal_outputs.loss:.4f}, "
#                       f"Plan similarity: {plan_similarity_loss:.4f}, "
#                       f"Random contrast: {random_contrast_loss:.4f}, "
#                       f"Total loss: {total_loss:.4f}")

#             normal_outputs.loss = total_loss

#         return normal_outputs

#     def calculate_plan_similarity_loss(self, normal_logits, plan_logits, human_end_positions, hidden_state_seq_len, plan_seq_len):
#         """方案1：分别比较各自的后续内容"""
#         losses = []
#         batch_size = normal_logits.size(0)

#         for i in range(batch_size):
#             pos = human_end_positions[i].item()
#             if pos < 0:
#                 continue

#             # 计算各自的结束位置
#             hidden_end_pos = pos + 1 + hidden_state_seq_len[i] + 1
#             plan_end_pos = pos + 1 + plan_seq_len[i] + 1
            
#             # 确定能比较的最大长度
#             normal_remaining = normal_logits.size(1) - hidden_end_pos
#             plan_remaining = plan_logits.size(1) - plan_end_pos
            
#             # 取较小的长度，确保比较的是相同长度的后续内容
#             compare_length = min(normal_remaining, plan_remaining, 200)
            
#             if compare_length <= 0:
#                 continue

#             # 分别从各自的结束位置开始提取
#             normal_logits_region = normal_logits[i, hidden_end_pos:hidden_end_pos + compare_length]
#             plan_logits_region = plan_logits[i, plan_end_pos:plan_end_pos + compare_length]

#             # 使用KL散度鼓励相似性（正常hidden state应该与plan文本相似）
#             normal_probs = F.softmax(normal_logits_region, dim=-1)
#             plan_probs = F.softmax(plan_logits_region.detach(), dim=-1)

#             # KL(normal || plan) - 鼓励normal分布接近plan分布
#             kl_loss = F.kl_div(
#                 F.log_softmax(normal_logits_region, dim=-1),
#                 plan_probs,
#                 reduction='batchmean'
#             )

#             # 添加余弦相似性作为辅助损失
#             cos_sim = F.cosine_similarity(
#                 normal_probs.view(-1),
#                 plan_probs.view(-1),
#                 dim=0
#             )
#             cos_loss = 1.0 - cos_sim

#             # 组合损失
#             combined_loss = 0.7 * kl_loss + 0.3 * cos_loss
#             losses.append(combined_loss)

#         if losses:
#             return torch.mean(torch.stack(losses))
#         return torch.tensor(0.0, device=normal_logits.device)

#     def calculate_random_contrast_loss(self, normal_logits, random_logits, human_end_positions, hidden_state_seq_len):
#         def js_divergence(p, q):
#             """计算JS散度"""
#             m = 0.5 * (p + q)
#             js_div = 0.5 * F.kl_div(p.log(), m, reduction='batchmean') + \
#                     0.5 * F.kl_div(q.log(), m, reduction='batchmean')
#             return js_div
        
#         losses = []
#         batch_size = normal_logits.size(0)
        
#         # 设定margin
#         margin = 0.5
        
#         for i in range(batch_size):
#             pos = human_end_positions[i].item()
#             if pos < 0:
#                 continue

#             # 计算插入内容之后的预测位置
#             hidden_end_pos = pos + 1 + hidden_state_seq_len[i] + 1

#             # 比较插入内容后的预测
#             compare_start = hidden_end_pos
#             compare_length = min(
#                 normal_logits.size(1) - compare_start,
#                 random_logits.size(1) - compare_start,
#                 200
#             )

#             if compare_length <= 0:
#                 continue

#             compare_end = compare_start + compare_length
#             normal_logits_region = normal_logits[i, compare_start:compare_end]
#             random_logits_region = random_logits[i, compare_start:compare_end]

#             normal_probs = F.softmax(normal_logits_region, dim=-1) + 1e-8  # 数值稳定性
#             random_probs = F.softmax(random_logits_region.detach(), dim=-1) + 1e-8

#             # 计算JS散度
#             js_div = js_divergence(normal_probs, random_probs)
            
#             # ✅ 只保留margin-based损失
#             contrast_loss = torch.max(
#                 torch.tensor(0.0, device=normal_logits.device),
#                 margin - js_div
#             )
            
#             # 可选：添加调试信息
#             if hasattr(self, '_debug_contrast') and self._debug_contrast and i == 0:
#                 print(f"JS散度: {js_div:.4f}, Margin: {margin}, 损失: {contrast_loss:.4f}")
            
#             losses.append(contrast_loss)

#         if losses:
#             return torch.mean(torch.stack(losses))
#         return torch.tensor(0.0, device=normal_logits.device)

#     # 其他保持不变的方法...
#     def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
#         self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

#     def get_input_embeddings(self):
#         return self.base_model.get_input_embeddings()

#     def set_input_embeddings(self, value):
#         self.base_model.set_input_embeddings(value)

#     def get_output_embeddings(self):
#         return self.base_model.get_output_embeddings()

#     def set_output_embeddings(self, new_embeddings):
#         self.base_model.set_output_embeddings(new_embeddings)

#     def resize_token_embeddings(self, new_num_tokens):
#         return self.base_model.resize_token_embeddings(new_num_tokens)

#     def prepare_inputs_for_generation(self, *args, **kwargs):
#         return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

#     @torch.no_grad()
#     def generate(self, *args, **kwargs):
#         return self.base_model.generate(*args, **kwargs)

#     def save_pretrained(self, save_directory, **kwargs):
#         """Save the model."""
#         # Save the base model
#         self.base_model.save_pretrained(save_directory, **kwargs)

#         # Save default prepended hidden state if learnable
#         if self.prepended_learnable:
#             prepended_state_path = os.path.join(save_directory, 'default_prepended_hidden_state.pt')
#             torch.save(self.default_prepended_hidden_state, prepended_state_path)

#         # Save configuration
#         prepended_config = {
#             'prepended_length': self.prepended_length,
#             'prepended_learnable': self.prepended_learnable,
#             'hidden_size': self.default_prepended_hidden_state.shape[-1],
#         }
#         config_path = os.path.join(save_directory, 'prepended_config.json')
#         with open(config_path, 'w') as f:
#             json.dump(prepended_config, f, indent=2)
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
IGNORE = -100
EPS = 1e-8  # 防止 log 0

class AdaptiveProjection(nn.Module):
    """自适应数值范围投影层"""

    def __init__(self, hidden_size):
        super().__init__()
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.output_scale = nn.Parameter(torch.tensor(0.1))

        # 动态范围适配层
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
        self._init_weights()

    def _init_weights(self):
        # 第一层：小范围初始化
        nn.init.normal_(self.proj[0].weight, mean=0, std=0.02)
        nn.init.zeros_(self.proj[0].bias)

        # 第二层：标准初始化
        nn.init.xavier_uniform_(self.proj[3].weight, gain=1e-2)
        nn.init.zeros_(self.proj[3].bias)

    def forward(self, x):
        # 三步处理流程
        residual = x * self.scale  # 保留原始缩放信号
        x = self.proj(residual)  # 特征变换
        return (residual + x) * self.output_scale  # 残差连接+校准


class ModelWithInsertedHiddenState(nn.Module):
    def __init__(self, base_model, prepended_length, hidden_size, prepended_learnable=False, prepended_input_dim=0, num_heads=8,
                 plan_similarity_weight=0.5, random_contrast_weight=1.5):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.step_count = 0
        self.prepended_length = prepended_length
        self.prepended_learnable = prepended_learnable
        self.config = base_model.config
        self.tokenizer = None
        self.ratio_list = []

        self.contrastive_weight = 0.1
        self.plan_similarity_weight = plan_similarity_weight
        self.random_contrast_weight = random_contrast_weight

        # MHA层配置
        self.hidden_mha = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                batch_first=True,
                dropout=0.1
            ).to(torch.bfloat16)
        
        self._init_mha_weights()

        # 归一化层
        self.pre_ln = nn.LayerNorm(hidden_size, eps=1e-6).to(torch.bfloat16)
        self.post_ln = nn.LayerNorm(hidden_size, eps=1e-6).to(torch.bfloat16)

        # 自适应投影层
        self.adaptive_proj = AdaptiveProjection(hidden_size).to(torch.bfloat16)

        # 输出处理层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GLU(dim=-1),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        ).to(torch.bfloat16)
        self._init_projection_weights()

        # 可学习缩放因子（从 adaptive_proj 获取）
        self.scale = self.adaptive_proj.scale  # ✅ 将 scale 暴露为当前类的属性
        self.output_scale = self.adaptive_proj.output_scale  # ✅ 同上

        self.input_projector = nn.Linear(prepended_input_dim, hidden_size, bias=True)

        # 如果需要learnable的默认hidden state（作为fallback）
        if prepended_learnable:
            self.default_prepended_hidden_state = nn.Parameter(
                torch.randn(prepended_length, hidden_size) * 0.02
            )
        else:
            self.register_buffer(
                'default_prepended_hidden_state',
                torch.zeros(prepended_length, hidden_size)
            )

    def _init_mha_weights(self):
        """初始化MHA层权重"""
        nn.init.xavier_uniform_(self.hidden_mha.in_proj_weight, gain=1.0 / math.sqrt(3))
        nn.init.xavier_uniform_(self.hidden_mha.out_proj.weight, gain=1.0)
        if self.hidden_mha.in_proj_bias is not None:
            nn.init.constant_(self.hidden_mha.in_proj_bias, 0.)
            nn.init.constant_(self.hidden_mha.out_proj.bias, 0.)

    def _init_projection_weights(self):
        """初始化输出投影层"""

        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1e-2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.output_projection.apply(init_fn)

    def process_hidden_states(self, hidden_states):
        """处理隐藏状态的核心流程"""
        # 类型转换保证精度
        orig_dtype = hidden_states.dtype
        # hidden_states = hidden_states.float()

        # 1. 预归一化
        normed = self.pre_ln(hidden_states)

        # 2. MHA处理
        attn_output, _ = self.hidden_mha(normed, normed, normed)
        attn_output = self.post_ln(normed + attn_output)  # 残差连接

        # 3. 自适应投影
        projected = self.adaptive_proj(attn_output)

        # 监控与调试
        if self.training and self.step_count % 100 == 0:
            self._log_stats(hidden_states, attn_output, projected)
        self.step_count += 1

        return projected.to(orig_dtype)

    def _log_stats(self, input, mha_out, projected):
        """记录训练统计信息"""
        stats = {
            '输入幅度': input.std().item(),
            'MHA输出幅度': mha_out.std().item(),
            '投影输出幅度': projected.std().item(),
        }
        print(" | ".join([f"{k}: {v:.3e}" for k, v in stats.items()]))

    def forward(self, input_tensors):
        """完整前向传播"""
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            processed = self.process_hidden_states(input_tensors)
            return torch.clamp(processed, -10.0, 10.0)

    # ✅ 新增：分布归一化方法
    def normalize_hidden_to_embedding_distribution(self, hidden_state, reference_embeds):
        """将hidden_state的分布调整为与reference_embeds相似"""
        # 计算reference_embeds的统计特性
        ref_mean = reference_embeds.mean()
        ref_std = reference_embeds.std()

        # 计算hidden_state的统计特性
        hidden_mean = hidden_state.mean()
        hidden_std = hidden_state.std()

        # Z-score标准化 + 重新缩放到reference分布
        normalized_hidden = (hidden_state - hidden_mean) / (hidden_std + 1e-8)
        adjusted_hidden = normalized_hidden * ref_std + ref_mean

        return adjusted_hidden

    # ✅ 新增：自适应混合方法
    def adaptive_mix_with_balanced_distribution(self, hidden_state, plan_embeds, mix_ratio):
        """平衡分布的自适应混合，增强数值稳定性"""

        # ✅ 输入检查
        if torch.isnan(hidden_state).any():
            print("WARNING: hidden_state contains NaN before mixing!")
            hidden_state = torch.nan_to_num(hidden_state, nan=0.0)

        if torch.isnan(plan_embeds).any():
            print("WARNING: plan_embeds contains NaN before mixing!")
            plan_embeds = torch.nan_to_num(plan_embeds, nan=0.0)

        # 计算统计信息，添加数值稳定性
        hidden_std = hidden_state.std() + 1e-8
        plan_std = plan_embeds.std() + 1e-8

        print(f"Debug - Before processing:")
        print(f"  Hidden std: {hidden_std:.6f}, Plan std: {plan_std:.6f}")
        print(f"  Ratio: {hidden_std / plan_std:.2f}")

        # ✅ 添加比率检查
        ratio = hidden_std / plan_std
        if torch.isnan(ratio) or torch.isinf(ratio):
            print("WARNING: Invalid ratio detected, using plan_embeds only")
            result = plan_embeds
        elif ratio > 3.0:  # 差异仍然很大时进行归一化
            hidden_state = self.normalize_hidden_to_embedding_distribution(hidden_state, plan_embeds)
            print(f"  Applied statistical normalization")

            # 重新计算ratio
            hidden_std = hidden_state.std() + 1e-8
            ratio = hidden_std / plan_std

        print(f"Hidden state 和 Plan 的混合比例为: {mix_ratio}, 前 {mix_ratio} 为 hidden_state")

        # 现在进行混合
        if mix_ratio == 0.0:
            result = plan_embeds
        elif mix_ratio == 1.0:
            result = hidden_state
        else:
            plan_len = plan_embeds.size(0)
            hidden_len = hidden_state.size(0)

            hidden_part_len = int(round(hidden_len * mix_ratio))
            plan_part_len = int(round(plan_len * mix_ratio))

            hidden_part = hidden_state[:hidden_part_len]
            plan_part = plan_embeds[plan_part_len:]

            result = torch.cat([hidden_part, plan_part], dim=0)

        # ✅ 输出检查
        # if torch.isnan(result).any():
        #     print("WARNING: Result contains NaN after mixing!")
        #     result = torch.nan_to_num(result, nan=0.0)

        print(f"Debug - After mixing:")
        print(f"  Result range: [{result.min():.6f}, {result.max():.6f}]")
        print(f"  Result std: {result.std():.6f}")

        return result

    def process_hidden_states_list(self, hidden_states_list):
        """对一个list中的所有hidden states进行MHA处理"""
        processed_list = []
        for hidden_state in hidden_states_list:
            if hidden_state is not None:
                # 对每个hidden state进行MHA处理
                processed = self.process_hidden_states(hidden_state.unsqueeze(0)).squeeze(0)
                processed_list.append(processed)
            else:
                processed_list.append(None)
        return processed_list

    def adjust_weights_dynamically(self, plan_similarity_loss, random_contrast_loss):
        """基于loss动态调整权重"""

        # Option 1: Use Python's built-in min/max functions for scalar values
        # random_contrast_loss: 0->0.1, 0.5->3.0
        contrast_normalized = max(0.0, min(1.0, random_contrast_loss / 0.5))
        self.random_contrast_weight = 0.1 + contrast_normalized * 1.9  # 0.1 + [0,1] * 2.9 = [0.1, 3.0]

        # plan_similarity_loss: 0->0.1, 1.0->2.0 (设1.0为上限，给超过0.8的情况留余量)
        plan_normalized = max(0.0, min(1.0, plan_similarity_loss / 1.0))
        self.plan_similarity_weight = 0.1 + plan_normalized * 1.9  # 0.1 + [0,1] * 1.9 = [0.1, 1.0]

    def insert_plan_tokens(self, input_ids, attention_mask, labels, human_end_positions, plans):
        """将plan文本插入到指定位置，前后加入特殊标记"""
        batch_size = input_ids.shape[0]
        new_input_ids = []
        new_attention_mask = []
        new_labels = []

        # 获取特殊标记的token ids
        bop_token_id = self.tokenizer.convert_tokens_to_ids('<bop>')
        eop_token_id = self.tokenizer.convert_tokens_to_ids('<eop>')

        for batch_idx in range(batch_size):
            if human_end_positions[batch_idx] >= 0 and plans[batch_idx]:
                insert_pos = human_end_positions[batch_idx].item()

                # 获取plan的token ids
                plan_token_ids = torch.tensor(plans[batch_idx], device=input_ids.device)

                # 分割原始序列
                before_ids = input_ids[batch_idx, :insert_pos]
                after_ids = input_ids[batch_idx, insert_pos:]

                # 创建带有特殊标记的plan序列: <bop> + plan + <eop>
                bop_tensor = torch.tensor([bop_token_id], device=input_ids.device)
                eop_tensor = torch.tensor([eop_token_id], device=input_ids.device)
                marked_plan_ids = torch.cat([bop_tensor, plan_token_ids, eop_tensor], dim=0)

                # 插入带标记的plan tokens
                new_seq = torch.cat([before_ids, marked_plan_ids, after_ids], dim=0)

                new_input_ids.append(new_seq)

                # 处理attention mask
                if attention_mask is not None:
                    before_mask = attention_mask[batch_idx, :insert_pos]
                    after_mask = attention_mask[batch_idx, insert_pos:]
                    plan_mask = torch.ones(len(marked_plan_ids), device=attention_mask.device)

                    new_mask = torch.cat([before_mask, plan_mask, after_mask], dim=0)

                    new_attention_mask.append(new_mask)

                # 处理labels
                if labels is not None:
                    before_labels = labels[batch_idx, :insert_pos]
                    after_labels = labels[batch_idx, insert_pos:]
                    plan_labels = torch.full((len(marked_plan_ids),), IGNORE_TOKEN_ID, device=labels.device)

                    new_label = torch.cat([before_labels, plan_labels, after_labels], dim=0)

                    new_labels.append(new_label)
            else:
                # 没有插入位置或plan，保持原样
                new_input_ids.append(input_ids[batch_idx])
                if attention_mask is not None:
                    new_attention_mask.append(attention_mask[batch_idx])
                if labels is not None:
                    new_labels.append(labels[batch_idx])

        result = {
            'input_ids': pad_sequence(new_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            'attention_mask': pad_sequence(new_attention_mask, batch_first=True,
                                           padding_value=0) if new_attention_mask else None,
            'labels': pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID) if new_labels else None
        }

        return result

    def generate_random_hidden_states(self, batch_size, device, dtype, hidden_state_seq_len):
        """生成更合理的随机hidden states"""
        hidden_states = []

        for i in range(batch_size):
            seq_len_i = hidden_state_seq_len[i]

            # 方案1: 从同一batch中随机选择其他样本的hidden state
            if len(hidden_state_seq_len) > 1:
                # 随机选择不同的索引
                other_indices = [j for j in range(batch_size) if j != i]
                if other_indices:
                    random_idx = random.choice(other_indices)
                    # 使用其他样本的hidden state作为"随机"对比
                    # 这样更有挑战性，因为都是真实的hidden state
                    if hasattr(self, '_current_batch_hidden_states') and self._current_batch_hidden_states:
                        other_hidden = self._current_batch_hidden_states[random_idx]
                        # 注意：这里不处理MHA，因为会在后面统一处理
                        # 调整长度匹配
                        if other_hidden.shape[0] >= seq_len_i:
                            hidden_state_i = other_hidden[:seq_len_i]
                        else:
                            # 如果长度不够，用其他样本的重复+噪声
                            repeat_times = (seq_len_i // other_hidden.shape[0]) + 1
                            repeated = other_hidden.repeat(repeat_times, 1)[:seq_len_i]
                            # 添加小噪声
                            noise = torch.randn_like(repeated) * 0.01
                            hidden_state_i = repeated + noise
                        hidden_states.append(hidden_state_i)
                        continue

            # 方案2: 改进的随机生成 - 模拟真实hidden state的分布
            # 使用更接近真实hidden state的统计特性
            hidden_state_i = torch.randn(
                seq_len_i, self.config.hidden_size,
                device=device, dtype=dtype
            )

            # 添加类似真实hidden state的结构
            # 1. 降低方差，真实hidden state通常不会有太大的值
            hidden_state_i = hidden_state_i * 0.1

            # 2. 添加一些周期性模式（模拟attention pattern）
            if seq_len_i > 4:
                for j in range(0, seq_len_i, 4):
                    end_j = min(j + 4, seq_len_i)
                    # 在小窗口内添加相关性
                    if j > 0:
                        hidden_state_i[j:end_j] = 0.7 * hidden_state_i[j:end_j] + 0.3 * hidden_state_i[
                                                                                        j - 1:j - 1 + end_j - j]

            # 3. 归一化到合理范围
            hidden_state_i = F.normalize(hidden_state_i, dim=-1) * 2.0

            hidden_states.append(hidden_state_i)

        return hidden_states

    def _forward_with_hidden_states(
            self, input_ids, attention_mask, inputs_embeds, labels,
            human_end_positions, prepended_hidden_states,  # 注意：这是 list of tensor
            past_key_values, use_cache, output_attentions,
            output_hidden_states, return_dict, **kwargs
    ):
        device = next(self.base_model.parameters()).device
        model_dtype = next(self.base_model.parameters()).dtype

        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids).to(model_dtype)

        inputs_embeds = inputs_embeds.to(dtype=model_dtype)

        # 如果提供了 prepended_hidden_states，则处理每个 batch 对应的 hidden state
        if prepended_hidden_states is not None:
            prepended_hidden_states = [h.to(dtype=model_dtype) for h in prepended_hidden_states]
            # 对所有hidden states进行MHA处理
            prepended_hidden_states = self.process_hidden_states_list(prepended_hidden_states)

        # 获取特殊标记的embeddings
        bop_token_id = self.tokenizer.convert_tokens_to_ids('<bop>')
        eop_token_id = self.tokenizer.convert_tokens_to_ids('<eop>')

        bop_embedding = self.base_model.get_input_embeddings()(torch.tensor([bop_token_id], device=device)).squeeze(0)
        eop_embedding = self.base_model.get_input_embeddings()(torch.tensor([eop_token_id], device=device)).squeeze(0)

        batch_size = inputs_embeds.shape[0]
        new_inputs_embeds = []
        new_attention_mask = [] if attention_mask is not None else None
        new_labels = [] if labels is not None else None

        for batch_idx in range(batch_size):
            insert_pos = human_end_positions[batch_idx].item()
            if insert_pos < 0:
                # 没有插入位置，保持原样
                new_inputs_embeds.append(inputs_embeds[batch_idx])
                if attention_mask is not None:
                    new_attention_mask.append(attention_mask[batch_idx])
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue

            # 获取原始输入嵌入
            before = inputs_embeds[batch_idx, :insert_pos]
            after = inputs_embeds[batch_idx, insert_pos:]

            # 获取当前 batch 的 prepended hidden state（已经经过MHA处理）
            if prepended_hidden_states is not None and batch_idx < len(prepended_hidden_states):
                hidden_state_to_use = prepended_hidden_states[batch_idx]
            # else:
            #     hidden_state_to_use = self.default_prepended_hidden_state

            # 在hidden state前后添加特殊标记embedding: <bop> + hidden_state + <eop>
            marked_hidden_state = torch.cat([
                bop_embedding.unsqueeze(0),  # <bop>
                hidden_state_to_use,  # hidden states (已经过MHA处理)
                eop_embedding.unsqueeze(0)  # <eop>
            ], dim=0).to(dtype=model_dtype)

            # 拼接
            batch_embeds = torch.cat([before, marked_hidden_state, after], dim=0)
            new_inputs_embeds.append(batch_embeds)

            # 处理 attention_mask
            if attention_mask is not None:
                before_mask = attention_mask[batch_idx, :insert_pos]
                after_mask = attention_mask[batch_idx, insert_pos:]
                prepended_mask = torch.ones(
                    marked_hidden_state.size(0),  # 包含特殊标记的长度
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                batch_mask = torch.cat([before_mask, prepended_mask, after_mask], dim=0)
                new_attention_mask.append(batch_mask)

            # 处理 labels
            if labels is not None:
                before_labels = labels[batch_idx, :insert_pos]
                after_labels = labels[batch_idx, insert_pos:]
                prepended_labels = torch.full(
                    (marked_hidden_state.size(0),),  # 包含特殊标记的长度
                    IGNORE_TOKEN_ID,
                    dtype=labels.dtype,
                    device=labels.device
                )
                batch_labels = torch.cat([before_labels, prepended_labels, after_labels], dim=0)
                new_labels.append(batch_labels)

        # Pad 到统一长度
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(new_inputs_embeds, batch_first=True, padding_value=0)
        if new_attention_mask:
            attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
        if new_labels:
            labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)

        return self.base_model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def _forward_with_hidden_states_curriculum(
            self, input_ids, plan_ids, attention_mask, inputs_embeds, labels,
            human_end_positions, prepended_hidden_states,  # 注意：这是 list of tensor
            past_key_values, use_cache, output_attentions,
            output_hidden_states, return_dict, **kwargs
    ):
        device = next(self.base_model.parameters()).device
        model_dtype = next(self.base_model.parameters()).dtype

        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids).to(model_dtype)

        plan_embeds_list = []

        for plan_item in plan_ids:
            plan_tensor = torch.tensor(plan_item, device=device)
            plan_embeds = self.base_model.get_input_embeddings()(plan_tensor).to(model_dtype)
            plan_embeds_list.append(plan_embeds)

        inputs_embeds = inputs_embeds.to(dtype=model_dtype)

        # 如果提供了 prepended_hidden_states，则处理每个 batch 对应的 hidden state
        if prepended_hidden_states is not None:
            prepended_hidden_states = [h.to(dtype=model_dtype) for h in prepended_hidden_states]
            # 对所有hidden states进行MHA处理
            prepended_hidden_states = self.process_hidden_states_list(prepended_hidden_states)

        # 获取特殊标记的embeddings
        bop_token_id = self.tokenizer.convert_tokens_to_ids('<bop>')
        eop_token_id = self.tokenizer.convert_tokens_to_ids('<eop>')

        bop_embedding = self.base_model.get_input_embeddings()(torch.tensor([bop_token_id], device=device)).squeeze(0)
        eop_embedding = self.base_model.get_input_embeddings()(torch.tensor([eop_token_id], device=device)).squeeze(0)

        batch_size = inputs_embeds.shape[0]
        new_inputs_embeds = []
        new_attention_mask = [] if attention_mask is not None else None
        new_labels = [] if labels is not None else None

        for batch_idx in range(batch_size):
            insert_pos = human_end_positions[batch_idx].item()
            if insert_pos < 0:
                # 没有插入位置，保持原样
                new_inputs_embeds.append(inputs_embeds[batch_idx])
                if attention_mask is not None:
                    new_attention_mask.append(attention_mask[batch_idx])
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue

            # 获取原始输入嵌入
            before = inputs_embeds[batch_idx, :insert_pos]
            after = inputs_embeds[batch_idx, insert_pos:]

            # 获取当前 batch 的 prepended hidden state（已经经过MHA处理）
            if prepended_hidden_states is not None and batch_idx < len(prepended_hidden_states):
                hidden_state_to_use = prepended_hidden_states[batch_idx]

            plan_embeds = plan_embeds_list[batch_idx]

            random_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

            if len(self.ratio_list) <= batch_idx:
                mix_ratio = random.choice(random_list)
                self.ratio_list.append(mix_ratio)
            else:
                mix_ratio = self.ratio_list[batch_idx]

            # mix_ratio = random.choice(random_list)

            # ✅ 使用改进的混合策略
            hidden_state_to_use = self.adaptive_mix_with_balanced_distribution(
                hidden_state_to_use, plan_embeds, mix_ratio
            )

            # 在hidden state前后添加特殊标记embedding: <bop> + hidden_state + <eop>
            marked_hidden_state = torch.cat([
                bop_embedding.unsqueeze(0),  # <bop>
                hidden_state_to_use,  # hidden states (已经过MHA处理)
                eop_embedding.unsqueeze(0)  # <eop>
            ], dim=0).to(dtype=model_dtype)

            # 拼接
            batch_embeds = torch.cat([before, marked_hidden_state, after], dim=0)
            new_inputs_embeds.append(batch_embeds)

            # 处理 attention_mask
            if attention_mask is not None:
                before_mask = attention_mask[batch_idx, :insert_pos]
                after_mask = attention_mask[batch_idx, insert_pos:]
                prepended_mask = torch.ones(
                    marked_hidden_state.size(0),  # 包含特殊标记的长度
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                batch_mask = torch.cat([before_mask, prepended_mask, after_mask], dim=0)
                new_attention_mask.append(batch_mask)

            # 处理 labels
            if labels is not None:
                before_labels = labels[batch_idx, :insert_pos]
                after_labels = labels[batch_idx, insert_pos:]
                prepended_labels = torch.full(
                    (marked_hidden_state.size(0),),  # 包含特殊标记的长度
                    IGNORE_TOKEN_ID,
                    dtype=labels.dtype,
                    device=labels.device
                )
                batch_labels = torch.cat([before_labels, prepended_labels, after_labels], dim=0)
                new_labels.append(batch_labels)

        # Pad 到统一长度
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(new_inputs_embeds, batch_first=True, padding_value=0)
        if new_attention_mask:
            attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
        if new_labels:
            labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)

        return self.base_model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            plans=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            human_end_positions=None,
            prepended_hidden_states=None,
            **kwargs
    ):
        device = next(self.base_model.parameters()).device
        model_dtype = next(self.base_model.parameters()).dtype
        self._current_batch_hidden_states = prepended_hidden_states
        print(f'prepended_hidden_states的个数: {len(prepended_hidden_states)}')

        # Move all inputs to the correct device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device)
        if labels is not None:
            labels = labels.to(device)
        if human_end_positions is not None:
            human_end_positions = human_end_positions.to(device)

        # Get batch size
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_len, hidden_size = inputs_embeds.shape
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # ========== 新增：生成三种情况的输出 ==========
        hidden_state_seq_len = []
        plan_seq_len = []
        for item in prepended_hidden_states:
            hidden_state_seq_len.append(item.shape[0])

        for item in plans:
            plan_seq_len.append(len(item))


        self.ratio_list = []
        normal_outputs = self._forward_with_hidden_states_curriculum(
            input_ids, plans, attention_mask, inputs_embeds, labels,
            human_end_positions, prepended_hidden_states,
            past_key_values, use_cache, output_attentions,
            output_hidden_states, return_dict, **kwargs
        )

        # 2. plan文本插入的输出（带特殊标记）
        plan_outputs = None
        if plans is not None:
            # 使用torch.no_grad()来防止这部分计算产生梯度
            with torch.no_grad():
                plan_data = self.insert_plan_tokens(
                    input_ids, attention_mask, labels, human_end_positions, plans
                )
                plan_outputs = self.base_model(
                    input_ids=plan_data['input_ids'],
                    attention_mask=plan_data['attention_mask'],
                    labels=plan_data['labels'],
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs
                )

        random_outputs = None
        # 3. 随机hidden state的输出（MHA会在_forward_with_hidden_states中处理）
        random_hidden_states = self.generate_random_hidden_states(
            batch_size, device, model_dtype, hidden_state_seq_len
        )

        with torch.no_grad():
            random_outputs = self._forward_with_hidden_states_curriculum(
                input_ids, plans, attention_mask, inputs_embeds, labels,
                human_end_positions, random_hidden_states,
                past_key_values, use_cache, output_attentions,
                output_hidden_states, return_dict, **kwargs
            )

        # ========== 计算KL散度损失 ==========
        if plan_outputs is not None and human_end_positions is not None:
            # 计算plan相似性损失（正常hidden state应该与plan相似）
            plan_similarity_loss = self.calculate_plan_similarity_loss(
                normal_outputs.logits, plan_outputs.logits, human_end_positions, hidden_state_seq_len, plan_seq_len
            )
            # 计算随机对比损失（正常hidden state应该与随机hidden state不同）
            random_contrast_loss = self.calculate_random_contrast_loss(
                normal_outputs.logits, random_outputs.logits, human_end_positions, hidden_state_seq_len
            )

            # 然后正常计算总loss
            total_loss = normal_outputs.loss + \
                         self.plan_similarity_weight * plan_similarity_loss + \
                         self.random_contrast_weight * random_contrast_loss

            # 打印损失信息用于调试
            if hasattr(self, 'training') and self.training:
                print(f"Normal loss: {normal_outputs.loss:.4f}, "
                      f"Plan similarity: {plan_similarity_loss:.4f}, "
                      f"Random contrast: {random_contrast_loss:.4f}, "
                      f"Total loss: {total_loss:.4f}")

            normal_outputs.loss = total_loss

        for name, param in self.hidden_mha.named_parameters():
            print(f"{name} - Mean: {param.data.mean():.6f}, Std: {param.data.std():.6f}")
        print(f"self.output_projection[0].weight.data.mean(){self.output_projection[0].weight.data.mean()}")
        print(f"self.output_projection[0].bias.data.mean(){self.output_projection[0].bias.data.mean()}")
        time.sleep(2)

        return normal_outputs

    def calculate_plan_similarity_loss(self, normal_logits, plan_logits, human_end_positions, hidden_state_seq_len,
                                       plan_seq_len):
        """修复版本：确保plan_logits不产生梯度"""
        losses = []
        batch_size = normal_logits.size(0)

        for i in range(batch_size):
            pos = human_end_positions[i].item()
            if pos < 0:
                continue

            hidden_end_pos = pos + 1 + hidden_state_seq_len[i] + 1
            plan_end_pos = pos + 1 + plan_seq_len[i] + 1

            normal_remaining = normal_logits.size(1) - hidden_end_pos
            plan_remaining = plan_logits.size(1) - plan_end_pos

            compare_length = min(normal_remaining, plan_remaining, 200)

            if compare_length <= 0:
                continue

            normal_logits_region = normal_logits[i, hidden_end_pos:hidden_end_pos + compare_length]
            plan_logits_region = plan_logits[i, plan_end_pos:plan_end_pos + compare_length]

            # ✅ 确保plan_logits_region不产生梯度（虽然它已经在no_grad下计算了）
            plan_logits_region = plan_logits_region.detach()  # 现在可以安全使用detach了

            normal_probs = F.softmax(normal_logits_region, dim=-1) + 1e-8
            plan_probs = F.softmax(plan_logits_region, dim=-1) + 1e-8

            # KL散度损失
            kl_loss = F.kl_div(
                F.log_softmax(normal_logits_region, dim=-1),
                plan_probs,
                reduction='batchmean'
            )

            # 余弦相似性损失
            cos_sim = F.cosine_similarity(
                normal_probs.view(-1),
                plan_probs.view(-1),
                dim=0
            )
            cos_loss = 1.0 - cos_sim

            combined_loss = 0.7 * kl_loss + 0.3 * cos_loss
            losses.append(combined_loss)

        if losses:
            return torch.mean(torch.stack(losses))
        return torch.tensor(0.0, device=normal_logits.device, requires_grad=True)

    def calculate_random_contrast_loss(self, normal_logits, random_logits, human_end_positions, hidden_state_seq_len):
        """修复版本：确保random_logits不产生梯度"""

        def js_divergence(p, q):
            m = 0.5 * (p + q)
            js_div = 0.5 * F.kl_div(p.log(), m, reduction='batchmean') + \
                     0.5 * F.kl_div(q.log(), m, reduction='batchmean')
            return js_div

        losses = []
        batch_size = normal_logits.size(0)
        margin = 0.5

        for i in range(batch_size):
            pos = human_end_positions[i].item()
            if pos < 0:
                continue

            hidden_end_pos = pos + 1 + hidden_state_seq_len[i] + 1
            compare_start = hidden_end_pos
            compare_length = min(
                normal_logits.size(1) - compare_start,
                random_logits.size(1) - compare_start,
                200
            )

            if compare_length <= 0:
                continue

            compare_end = compare_start + compare_length
            normal_logits_region = normal_logits[i, compare_start:compare_end]
            random_logits_region = random_logits[i, compare_start:compare_end]

            # ✅ 确保random_logits_region不产生梯度
            random_logits_region = random_logits_region.detach()

            normal_probs = F.softmax(normal_logits_region, dim=-1) + 1e-8
            random_probs = F.softmax(random_logits_region, dim=-1) + 1e-8

            js_div = js_divergence(normal_probs, random_probs)

            contrast_loss = torch.max(
                torch.tensor(0.0, device=normal_logits.device),
                margin - js_div
            )

            losses.append(contrast_loss)

        if losses:
            return torch.mean(torch.stack(losses))
        return torch.tensor(0.0, device=normal_logits.device, requires_grad=True)

    # 其他保持不变的方法...
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.base_model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens):
        return self.base_model.resize_token_embeddings(new_num_tokens)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        """修复版本：确保保存时不影响梯度"""
        os.makedirs(save_directory, exist_ok=True)

        # Save the base model
        self.base_model.save_pretrained(save_directory, **kwargs)

        # Save default prepended hidden state if learnable
        if self.prepended_learnable:
            prepended_state_path = os.path.join(save_directory, 'default_prepended_hidden_state.pt')
            torch.save(self.default_prepended_hidden_state, prepended_state_path)

        # ✅ 修复MHA state保存
        mha_state_path = os.path.join(save_directory, 'hidden_mha_state.pt')
        mha_state = {
            'hidden_mha': self.hidden_mha.state_dict(),
            'pre_ln': self.pre_ln.state_dict(),
            'post_ln': self.post_ln.state_dict(),
            # 'post_mha_ln': self.post_mha_ln.state_dict(),
            'output_projection': self.output_projection.state_dict(),
            'scale': self.scale.item(),          
            'output_scale': self.output_scale.item(), 
        }
        torch.save(mha_state, mha_state_path)

        # 配置保存
        prepended_config = {
            'prepended_length': self.prepended_length,
            'prepended_learnable': self.prepended_learnable,
            'hidden_size': self.default_prepended_hidden_state.shape[-1],
            'mha_num_heads': self.hidden_mha.num_heads,
            'plan_similarity_weight': self.plan_similarity_weight,
            'random_contrast_weight': self.random_contrast_weight,
        }
        config_path = os.path.join(save_directory, 'prepended_config.json')
        with open(config_path, 'w') as f:
            json.dump(prepended_config, f, indent=2)

        print(f"✅ Model saved.")

    # 3. ✅ 额外检查：确保所有参数正确设置requires_grad
    def verify_gradient_requirements(self):
        """验证所有自定义参数的梯度设置"""
        print("🔍 验证梯度设置:")

        # 检查MHA参数
        mha_total = 0
        mha_requires_grad = 0
        for name, param in self.hidden_mha.named_parameters():
            mha_total += 1
            if param.requires_grad:
                mha_requires_grad += 1
            print(f"  MHA.{name}: requires_grad={param.requires_grad}, is_leaf={param.is_leaf}")

        print(f"  MHA参数总数: {mha_total}, 需要梯度: {mha_requires_grad}")

        # 检查投影层参数
        proj_total = 0
        proj_requires_grad = 0
        for i, layer in enumerate(self.output_projection):
            if hasattr(layer, 'weight'):
                proj_total += 1
                if layer.weight.requires_grad:
                    proj_requires_grad += 1
                print(f"  Proj[{i}].weight: requires_grad={layer.weight.requires_grad}, is_leaf={layer.weight.is_leaf}")
            if hasattr(layer, 'bias') and layer.bias is not None:
                proj_total += 1
                if layer.bias.requires_grad:
                    proj_requires_grad += 1
                print(f"  Proj[{i}].bias: requires_grad={layer.bias.requires_grad}, is_leaf={layer.bias.is_leaf}")

        print(f"  投影层参数总数: {proj_total}, 需要梯度: {proj_requires_grad}")
