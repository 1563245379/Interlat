import json
import time
import logging
from typing import List, Dict, Union, Any, Optional

from fastchat.model.model_adapter import get_conversation_template  # 现在其实不用 conv 了，但保留 import 不影响
from requests.exceptions import Timeout, ConnectionError

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import argparse

from .base import LMAgent

logger = logging.getLogger("agent_frame")


# =======================
# 包装模型（目前不做真正的 plan 注入，仅作一层壳）
# =======================
class ModelWithPlanInjection(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        plans=None,          # 先保留这个入参，但不在 forward 里用
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        human_end_positions=None,
        **kwargs
    ):
        # 计划注入逻辑在外面通过 inputs_embeds / prompt 文本完成
        return self.base_model(
            input_ids=input_ids,
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

    # 下面这些接口直接透传给 base_model，方便 generate/resize 等操作
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

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        self.base_model.save_pretrained(save_directory, **kwargs)


def set_model(model_path: str, device: str):
    """
    加载训练好的模型和 tokenizer（plan-only 模型）。
    """
    special_tokens = ['<FIRST_HUMAN_END>', '<bop>', '<eop>']

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    model = ModelWithPlanInjection(base_model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
    )

    # 如果你确认导出的模型已经包含这些 special tokens，可以保持注释；
    # 如果不放心，可以打开下面几行：
    # tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    # model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    print(f"Loaded model and tokenizer from {model_path}")
    return model, tokenizer


# 一个简单的 embedding helper，复用你 HiddenAgent 的写法
def get_embedding(text: str, model, tokenizer, device: str):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(input_ids).to(torch.bfloat16)
    return embeddings


class PlanAgent(LMAgent):
    """
    只做 plan 注入（文本版本：在 prompt 里用 <bop>plan<eop>），不插 hidden state。
    使用与 HiddenAgent 相同的 template 拼接方式（Qwen / LLaMA3 风格）。
    """

    def __init__(
        self,
        config,
        MODEL: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self.model_name = config["model_name"]
        if MODEL is None:
            # Default plan-only checkpoint - replace with your model path
            self.model_name = "/path/to/your/plan/model"
        else:
            self.model_name = MODEL

        print(f"[PlanAgent] using {self.model_name}.")

        self.temperature = config.get("temperature", 0.8)
        self.max_new_tokens = 100
        self.top_p = config.get("top_p", 1.0)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model, self.tokenizer = set_model(self.model_name, self.device)
        self.embedded_messages: List[Dict[str, torch.Tensor]] = []

        # 粗略根据路径判断模型家族，以选择 ChatML / LLaMA3 模板
        name_lower = self.model_name.lower()
        if "qwen" in name_lower:
            self.model_family = "qwen"
        elif "llama" in name_lower or "meta-llama" in name_lower:
            self.model_family = "llama"
        else:
            self.model_family = "unknown"  # 默认走 qwen 的 ChatML 风格

    def __call__(self, messages: List[dict], plan: str) -> str:
        """
        messages: [{"role": "user"/"assistant", "content": str}, ...]
        plan: 纯文本的 step-by-step plan，会被插在第一条 user 输入的 '<bop>' 和 '<eop>' 之间
        """
        # 生成参数（主要给日志留着）
        gen_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "echo": False,
            "top_p": self.top_p,
        }

        # 如果会话长度比缓存的消息短，说明是新对话，重置
        if len(messages) < len(self.embedded_messages):
            self.embedded_messages = []

        # 根据家族决定 EOT token
        eos_token_id = None

        # 按条目构建 template + embedding
        for i, item in enumerate(messages):
            role = item["role"]
            content = item["content"]

            # =============== 按模型家族拼接 template ===============
            # Qwen / ChatML 风格
            if self.model_family in ["qwen", "unknown"]:
                EOT = "<|im_end|>"
                eos_token_id = self.tokenizer.convert_tokens_to_ids(EOT)

                if role == "user" and isinstance(content, str) and i >= len(self.embedded_messages):
                    if i == 0:
                        # 第一条 user：带上 system + plan
                        content = (
                            "<|im_start|>system\n"
                            "You are a helpful assistant."
                            "<|im_end|>\n"
                            "<|im_start|>user\n"
                            f"{content}\n"
                            "Now, you are given a step-by-step plan to complete this task as follow: "
                            "<bop>"
                            f"{plan}"
                            "<eop>"
                            "<|im_end|>\n"
                            "<|im_start|>assistant\n"
                        )
                    else:
                        # 后续 user：正常 ChatML 拼一轮
                        content = (
                            "<|im_start|>user\n"
                            + content
                            + "<|im_end|>\n"
                            "<|im_start|>assistant\n"
                        )
                elif role == "assistant" and isinstance(content, str) and i >= len(self.embedded_messages):
                    content = content + "<|im_end|>\n"

            # LLaMA3 头格式
            elif self.model_family == "llama":
                USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n"
                ASSIST_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n"
                EOT = "<|eot_id|>"
                eos_token_id = self.tokenizer.convert_tokens_to_ids(EOT)

                if role == "user" and isinstance(content, str) and i >= len(self.embedded_messages):
                    if i == 0:
                        # 第一条 user：在内容后插入 plan，然后 EOT，然后 assistant header
                        content = (
                            USER_HEADER
                            + content
                            + EOT
                            + ASSIST_HEADER
                        )
                    else:
                        # 后续 user：一轮完整 user -> EOT -> assistant header
                        content = (
                            USER_HEADER
                            + content
                            + EOT
                            + ASSIST_HEADER
                        )
                elif role == "assistant" and isinstance(content, str) and i >= len(self.embedded_messages):
                    # assistant 消息后面补一个 EOT
                    content = ASSIST_HEADER + content + EOT + "\n"

            # =============== 计算 embedding ===============
            if i >= len(self.embedded_messages):
                embedded_content = get_embedding(
                    content, self.model, self.tokenizer, self.device
                ).to(torch.bfloat16)

                self.embedded_messages.append(
                    {
                        "role": role,
                        "embedding": embedded_content,
                    }
                )

        # 拼接所有消息的 embedding
        embeddings = []
        for msg in self.embedded_messages:
            emb = msg["embedding"]
            if emb.device != self.device:
                emb = emb.to(self.device)
            embeddings.append(emb)

        combined_embedding = torch.cat(embeddings, dim=1)  # [1, total_seq_len, hidden]

        # attention mask，全 1（因为目前没有 padding）
        mask_list = []
        for msg in self.embedded_messages:
            emb = msg["embedding"]
            seq_len = emb.size(1)
            mask = torch.ones(seq_len, dtype=torch.long, device=self.device)
            mask_list.append(mask)
        attention_mask = torch.cat(mask_list, dim=0).unsqueeze(0).to(self.device)

        # 兜底：如果没成功找到 eos_token_id，就用 tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": True,
        }

        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=combined_embedding,
                attention_mask=attention_mask,
                eos_token_id=eos_token_id,
                **gen_kwargs,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 如果你希望只保留第一个 Action 段，保留你原来的截断逻辑
        action_idx = full_text.find("\nAction:")
        if action_idx != -1:
            next_newline = full_text.find("\n", action_idx + 1)
            if next_newline != -1:
                return full_text[:next_newline]
            else:
                return full_text
        else:
            return full_text
