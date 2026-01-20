import json
import time
import logging
from typing import List, Dict, Union, Any, Optional
from fastchat.model.model_adapter import get_conversation_template
from requests.exceptions import Timeout, ConnectionError
import os
import gc
import warnings

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import argparse

from .base import LMAgent
from .model_insert_hidden import ModelWithInsertedHiddenState
from .align_block import AlignBlock

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用 GPU 2

logger = logging.getLogger("agent_frame")


def _add_to_set(s, new_stop):
    if not s:
        return
    if isinstance(s, str):
        new_stop.add(s)
    else:
        new_stop.update(s)


# -----------------------------
# 通用工具
# -----------------------------
def _device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def _attn_impl_for_device(device: str) -> Optional[str]:
    # 只有 CUDA 环境可安全启用 FA2
    return "flash_attention_2" if device.startswith("cuda") else None


def _clean_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """去掉常见前缀 'module.' 和 'model.'，以适配 Trainer/DDP 保存命名。"""
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("model."):
            k = k[6:]
        new_sd[k] = v
    return new_sd


def _detect_source_type(path_or_id: str) -> str:
    """
    返回：
      - 'hf_export'：目录下含 hidden_mha_state.pt（你自定义模块的权重单独导出）
      - 'trainer_ckpt'：目录下 pytorch_model.bin 里包含 hidden_mha（wrapper 的完整 state_dict）
      - 'base_only'：其他情况（本地基础模型目录或 Hub 名称）
    """
    if os.path.isdir(path_or_id):
        # 明确的 HF 导出
        if os.path.exists(os.path.join(path_or_id, "hidden_mha_state.pt")):
            return "hf_export"

        # 可能是 checkpoint-XXXX
        pm = os.path.join(path_or_id, "pytorch_model.bin")
        if os.path.exists(pm):
            try:
                # 仅 CPU 上读取 keys，避免显存占用
                raw = torch.load(pm, map_location="cpu")
                if isinstance(raw, dict) and "state_dict" in raw:
                    raw = raw["state_dict"]
                keys = list(raw.keys())
                # 任一 key 包含 hidden_mha 基本可判定为 trainer 中间 ckpt
                if any("hidden_mha" in k for k in keys):
                    return "trainer_ckpt"
            except Exception:
                # 读取失败按 base_only 处理
                pass

        # 其余当做基础模型目录
        return "base_only"

    # 非目录（大概率是 Hub 名称）
    return "base_only"


def _ensure_special_tokens_and_resize(tokenizer, model, tokens: List[str]):
    """如果 tokenizer 缺少某些特殊 token，就添加并 resize 模型 embedding。"""
    to_add = []
    for t in tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is None or tid == tokenizer.unk_token_id:
            to_add.append(t)
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        # 保留已有权重，新增行随机初始化
        model.resize_token_embeddings(len(tokenizer))


def _load_tokenizer_fallback(dir_or_id: str, config: Optional[AutoConfig]):
    """
    优先从本地目录加载；不行则尝试从 config._name_or_path 回退；再不行抛错。
    """
    # 1) 尝试目录/Hub
    try:
        return AutoTokenizer.from_pretrained(dir_or_id, trust_remote_code=True, use_fast=False)
    except Exception:
        pass

    # 2) 从 config 里找 _name_or_path
    base_name = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
    if base_name:
        try:
            return AutoTokenizer.from_pretrained(base_name, trust_remote_code=True, use_fast=False)
        except Exception:
            pass

    # 3) 最后兜底：你训练时常用的基座（可按需改）
    fallback = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        return AutoTokenizer.from_pretrained(fallback, trust_remote_code=True, use_fast=False)
    except Exception:
        # 彻底失败
        raise RuntimeError(
            f"无法加载 tokenizer。尝试过：\n"
            f"  - {dir_or_id}\n"
            f"  - {base_name}\n"
            f"  - {fallback}\n"
        )


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def get_memory_size(module, dtype=torch.float32):
    """计算模块的内存占用（MB）"""
    param_count = count_parameters(module)
    bytes_per_param = 4 if dtype == torch.float32 else 2  # bf16/fp16 = 2 bytes
    return param_count * bytes_per_param / (1024 * 1024)


def load_adaptive_proj_from_deepspeed(
    model,
    ckpt_path: str,
    cache_path: Optional[str] = None,
):
    """
    从 DeepSpeed 的 mp_rank_00_model_states.pt 中恢复
    - model.adaptive_proj.* 参数
    - model.scale / model.output_scale（如果在 ckpt 里有）

    同时：
    - 优先使用 cache_path（小文件）加速下次加载
    - 如果 cache_path 不存在，则从大 ckpt 抽取并保存
    """
    # 1) 先看有没有缓存的小文件
    if cache_path is None:
        base_dir = os.path.dirname(ckpt_path)
        cache_path = os.path.join(base_dir, "adaptive_proj_state.pt")

    if cache_path and os.path.exists(cache_path):
        print(f"[adaptive_proj] Found cached state at: {cache_path}, loading...")
        cached = torch.load(cache_path, map_location="cpu")

        # 恢复 adaptive_proj 参数
        if "adaptive_proj" not in cached:
            raise RuntimeError(f"[adaptive_proj] Cached file {cache_path} missing 'adaptive_proj' key.")
        adaptive_state = cached["adaptive_proj"]
        missing, unexpected = model.adaptive_proj.load_state_dict(adaptive_state, strict=False)
        print(f"[adaptive_proj] (cached) load_state_dict done. missing={missing}, unexpected={unexpected}")

        # 恢复 scale/output_scale（如果存在）
        if hasattr(model, "scale") and ("scale" in cached):
            with torch.no_grad():
                model.scale.data.copy_(cached["scale"].to(dtype=model.scale.dtype))
            print("[adaptive_proj] (cached) Restored model.scale")

        if hasattr(model, "output_scale") and ("output_scale" in cached):
            with torch.no_grad():
                model.output_scale.data.copy_(cached["output_scale"].to(dtype=model.output_scale.dtype))
            print("[adaptive_proj] (cached) Restored model.output_scale")

        return  # 直接返回

    # 2) 没有缓存，只能从 Deepspeed 大文件里读
    print(f"[adaptive_proj] Cache not found, loading from Deepspeed checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    # 找真正的 state_dict
    if isinstance(state, dict) and "module" in state:
        sd = state["module"]
        print("[adaptive_proj] Using state['module'] as state_dict")
    else:
        sd = state
        print("[adaptive_proj] No 'module' key, treat whole file as state_dict")

    # 过滤出 adaptive_proj.* 参数
    adaptive_state = {}
    for k, v in sd.items():
        if k.startswith("adaptive_proj."):
            new_key = k[len("adaptive_proj."):]  # 'proj.0.weight' 这类 key
            adaptive_state[new_key] = v

    if not adaptive_state:
        raise RuntimeError(
            f"[adaptive_proj] No keys starting with 'adaptive_proj.' found in {ckpt_path}"
        )

    print(f"[adaptive_proj] Found {len(adaptive_state)} params, loading into model.adaptive_proj ...")
    missing, unexpected = model.adaptive_proj.load_state_dict(adaptive_state, strict=False)
    print(f"[adaptive_proj] load_state_dict done. missing={missing}, unexpected={unexpected}")

    # 恢复 scale / output_scale（如果存在）
    scale_tensor = None
    out_scale_tensor = None

    if hasattr(model, "scale") and "scale" in sd:
        scale_tensor = sd["scale"]
        with torch.no_grad():
            model.scale.data.copy_(scale_tensor.to(dtype=model.scale.dtype))
        print("[adaptive_proj] Restored model.scale from Deepspeed ckpt")

    if hasattr(model, "output_scale") and "output_scale" in sd:
        out_scale_tensor = sd["output_scale"]
        with torch.no_grad():
            model.output_scale.data.copy_(out_scale_tensor.to(dtype=model.output_scale.dtype))
        print("[adaptive_proj] Restored model.output_scale from Deepspeed ckpt")

    # 3) 把提取出来的东西单独缓存到小文件，方便下次快速加载
    if cache_path:
        to_save = {
            "adaptive_proj": adaptive_state,
        }
        if scale_tensor is not None:
            to_save["scale"] = scale_tensor
        if out_scale_tensor is not None:
            to_save["output_scale"] = out_scale_tensor

        try:
            torch.save(to_save, cache_path)
            print(f"[adaptive_proj] Cached adaptive_proj state to: {cache_path}")
        except Exception as e:
            warnings.warn(f"[adaptive_proj] Failed to save cache to {cache_path}: {e}")


import os
import json
import gc
from collections import OrderedDict
import torch
from transformers import AutoModelForCausalLM

# -----------------------------
# 从 Deepspeed ckpt 中抽取 input_projector 的小工具
# -----------------------------
def _extract_input_projector_from_mp(mp_sd) -> OrderedDict:
    """
    从 mp_rank_00_model_states.pt 的内容中提取 input_projector 的权重。
    支持两种常见结构：
    1) 顶层就有 key "input_projector"，且对应的是子 state_dict
    2) 顶层有 "module"/"model"/"state_dict"，内部是一个完整 state_dict，
       其中参数名以 "input_projector." 开头。
    """
    if not isinstance(mp_sd, dict):
        return OrderedDict()

    # 情况 1：顶层直接有 input_projector 子字典
    if "input_projector" in mp_sd and isinstance(mp_sd["input_projector"], dict):
        print("[hf_export] found 'input_projector' dict at top-level in mp_rank_00_model_states.pt")
        return OrderedDict(mp_sd["input_projector"])

    # 情况 2：从 module / model / state_dict 里按前缀提取
    candidate_keys = ["module", "model", "state_dict"]
    for ck in candidate_keys:
        if ck in mp_sd and isinstance(mp_sd[ck], dict):
            base_sd = mp_sd[ck]
            sub = OrderedDict()
            for name, param in base_sd.items():
                if not isinstance(param, torch.Tensor):
                    continue
                if name.startswith("input_projector."):
                    new_name = name[len("input_projector."):]
                    sub[new_name] = param
            if len(sub) > 0:
                print(f"[hf_export] extracted input_projector from '{ck}' in mp_rank_00_model_states.pt "
                      f"with {len(sub)} tensors")
                return sub

    return OrderedDict()


def set_model(MODEL: str, cross_family=False):
    """
    统一加载：
      - HF 导出目录（含 hidden_mha_state.pt）
      - Trainer checkpoint（checkpoint-XXXX/，pytorch_model.bin 是 wrapper 的完整 state_dict）
      - 仅基础模型（本地目录或 Hub 名称）
    """
    source_type = _detect_source_type(MODEL)
    device, torch_dtype = _device_and_dtype()
    attn_impl = _attn_impl_for_device(device)

    print(f"[set_model] source_type={source_type}, device={device}, dtype={torch_dtype}")

    if source_type == "hf_export":
        # --- 情况 1：你最终导出的 HF 风格目录 ---
        # 1) 直接把 base 模型按 HF 规范加载
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        ).to(device)

        # 2) tokenizer — 从同目录加载
        tokenizer = _load_tokenizer_fallback(MODEL, base_model.config)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = "right"

        # 3) 包上 wrapper
        hidden_size = base_model.config.hidden_size
        # prepended_length：优先从 prepended_config.json 读取
        prep_len = 800
        cfg_path = os.path.join(MODEL, "prepended_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                j = json.load(f)
            prep_len = int(j.get("prepended_length", prep_len))

        model = ModelWithInsertedHiddenState(
            base_model=base_model,
            prepended_length=prep_len,
            hidden_size=hidden_size,
            prepended_learnable=False,
            prepended_input_dim=3584,
        ).to(device)

        # 4) 读自定义层权重
        mha_state_path = os.path.join(MODEL, "hidden_mha_state_small.pt")
        if not os.path.exists(mha_state_path):
            print('不存在 hidden_mha_state_small.pt，回退到 hidden_mha_state.pt')
            mha_state_path = os.path.join(MODEL, "hidden_mha_state.pt")
        else:
            print('存在 hidden_mha_state_small.pt')
        mha_state = torch.load(mha_state_path, map_location="cpu")

        # 基本模块
        model.hidden_mha.load_state_dict(mha_state["hidden_mha"])
        model.pre_ln.load_state_dict(mha_state["pre_ln"])
        model.post_ln.load_state_dict(mha_state["post_ln"])

        # adaptive_proj: 新版 hidden_mha_state 已经有；老版没有时用 Deepspeed 恢复
        if "adaptive_proj" in mha_state:
            print("[hf_export] Found adaptive_proj in hidden_mha_state, loading directly.")
            model.adaptive_proj.load_state_dict(mha_state["adaptive_proj"])
        else:
            print("[hf_export] adaptive_proj NOT found in hidden_mha_state, "
                  "fallback to Deepspeed mp_rank_00_model_states.pt ...")
            ds_ckpt_path = os.path.join(MODEL, "mp_rank_00_model_states.pt")
            cache_path = os.path.join(MODEL, "adaptive_proj_state.pt")
            load_adaptive_proj_from_deepspeed(model, ds_ckpt_path, cache_path=cache_path)

        # ---------- 新增：input_projector 加载逻辑 ----------
        if hasattr(model, "input_projector") and cross_family:
            input_proj_loaded = False

            # 1) 优先从 hidden_mha_state_small.pt / hidden_mha_state.pt 中加载
            if "input_projector" in mha_state:
                try:
                    print("[hf_export] Found input_projector in hidden_mha_state, loading directly.")
                    model.input_projector.load_state_dict(mha_state["input_projector"])
                    input_proj_loaded = True
                except Exception as e:
                    print(f"[hf_export] Failed to load input_projector from hidden_mha_state: {e}")

            # 2) 如果没有或加载失败，则尝试从 Deepspeed ckpt 中提取
            if not input_proj_loaded:
                ds_ckpt_path = os.path.join(MODEL, "mp_rank_00_model_states.pt")
                if os.path.exists(ds_ckpt_path):
                    print("[hf_export] input_projector not in hidden_mha_state, "
                          "trying Deepspeed mp_rank_00_model_states.pt ...")
                    try:
                        mp_sd = torch.load(ds_ckpt_path, map_location="cpu")
                        extracted = _extract_input_projector_from_mp(mp_sd)
                        if len(extracted) > 0:
                            model.input_projector.load_state_dict(extracted)
                            input_proj_loaded = True
                            print("[hf_export] Successfully loaded input_projector from mp_rank_00_model_states.pt")
                        else:
                            print("[hf_export] No input_projector found in mp_rank_00_model_states.pt, "
                                  "treat as no input_projector.")
                    except Exception as e:
                        print(f"[hf_export] Failed to load input_projector from Deepspeed checkpoint: {e}")
                else:
                    print("[hf_export] mp_rank_00_model_states.pt not found, "
                          "treat as no input_projector.")
        else:
            print("[hf_export] model has no attribute 'input_projector', skip loading it.")
        # ---------- 新增部分结束 ----------

        # scale / output_scale：如果 hidden_mha_state 里有，就覆盖一次
        scale_val = mha_state.get("scale", None)
        out_scale_val = mha_state.get("output_scale", None)
        if scale_val is not None and hasattr(model, "scale"):
            with torch.no_grad():
                model.scale.data.copy_(torch.tensor(scale_val, dtype=model.scale.dtype))
        if out_scale_val is not None and hasattr(model, "output_scale"):
            with torch.no_grad():
                model.output_scale.data.copy_(torch.tensor(out_scale_val, dtype=model.output_scale.dtype))

        # 5) 特殊 token（若缺少则补齐）
        _ensure_special_tokens_and_resize(tokenizer, model.base_model,
                                          ['<FIRST_HUMAN_END>', '<bop>', '<eop>'])

        # 统计信息
        mha_params = count_parameters(model.hidden_mha)
        mha_size = get_memory_size(model.hidden_mha, torch_dtype)
        print(f"MHA层参数量: {mha_params:,} ({mha_size:.1f}MB)")

        proj_params = count_parameters(model.adaptive_proj)
        proj_size = get_memory_size(model.adaptive_proj, torch_dtype)
        print(f"AdaptiveProjection参数量: {proj_params:,} ({proj_size:.1f}MB)")

        ln_params = count_parameters(model.pre_ln) + count_parameters(model.post_ln)
        ln_size = (get_memory_size(model.pre_ln, torch_dtype) +
                   get_memory_size(model.post_ln, torch_dtype))
        print(f"LayerNorm层参数量: {ln_params:,} ({ln_size:.1f}MB)")

        if model.input_projector is not None:
            model.input_projector.to(device).to(torch.bfloat16)

        # 如果你想，也可以在这里打印 input_projector 的规模
        if hasattr(model, "input_projector"):
            try:
                ip_params = count_parameters(model.input_projector)
                ip_size = get_memory_size(model.input_projector, torch_dtype)
                print(f"InputProjector层参数量: {ip_params:,} ({ip_size:.1f}MB)")
            except Exception:
                pass

        del mha_state
        gc.collect()
        torch.cuda.empty_cache()

        return model, tokenizer

# =========================
# 其余代码基本不动
# =========================

def get_embedding(text: str, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(input_ids).to(device)
    return embeddings

class HiddenAgent(LMAgent):
    """This agent is a test agent, which does nothing. (return empty string for each action)"""

    def __init__(
        self,
        config,
        MODEL,
        cross_family=False
    ) -> None:
        super().__init__(config)
        self.model_name = config["model_name"]
        self.temperature = config.get("temperature", 0.8)
        self.max_new_tokens = 100
        self.top_p = config.get("top_p", 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedded_messages = []
        self.model, self.tokenizer = set_model(MODEL, cross_family)

        # 根据 model_name 简单判一下家族
        name_lower = self.model_name.lower()
        if "qwen" in name_lower:
            self.model_family = "qwen"
        elif "llama" in name_lower or "meta-llama" in name_lower:
            # 这里默认按 LLaMA3 风格来硬编码
            self.model_family = "llama"
        else:
            self.model_family = "unknown"  # 默认用 qwen 样式也行，看你喜好

        # 添加注意力分析器
        self.attention_analyzer = None

    def enable_attention_analysis(self):
        """启用注意力分析功能"""
        self.attention_analyzer = EmbeddingAttentionAnalyzer(self.model, self.tokenizer)
        print("Attention analysis enabled!")

    def __call__(self, messages: List[dict], hidden_state) -> str:
        gen_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "echo": False,
            "top_p": self.top_p,
        }

        hidden_state = hidden_state.to(self.device).to(torch.bfloat16)

        # if self.model.input_projector is not None:
        #     hidden_state = self.model.input_projector(hidden_state)

        # 你自定义的 MHA + adaptive_proj
        normed = self.model.pre_ln(hidden_state)
        attn_output, _ = self.model.hidden_mha(normed, normed, normed)
        attn_output = self.model.post_ln(normed + attn_output)
        hidden_state = self.model.adaptive_proj(attn_output)

        conv = get_conversation_template(self.model_name)
        for history_item in messages:
            role = history_item["role"]
            content = history_item["content"]
            if role == "user":
                conv.append_message(conv.roles[0], content)
            elif role == "assistant":
                conv.append_message(conv.roles[1], content)
            else:
                raise ValueError(f"Unknown role: {role}")
        conv.append_message(conv.roles[1], None)

        input_embeds = None

        if len(messages) < len(self.embedded_messages):
            # 会话重置
            self.embedded_messages = []

        for i, item in enumerate(messages):
            role = item['role']
            content = item['content']

            # =============== 按模型家族硬编码文本内容 ===============
            # Qwen 风格
            if self.model_family == "qwen" or self.model_family == "unknown":
                EOT = '<|im_end|>'
                eos_token_id = self.tokenizer.convert_tokens_to_ids(EOT)
                if role == 'user' and isinstance(content, str) and i >= len(self.embedded_messages):
                    if i == 0:
                        content = (
                            '<|im_start|>system\n'
                            "You are a helpful assistant."
                            + '<|im_end|>\n'
                            + '<|im_start|>user\n'
                            + content
                            + 'Now, you are given a step-by-step plan to complete this task as follow: '
                            + '<bop>'
                        )
                    else:
                        content = (
                            '<|im_start|>user\n'
                            + content
                            + '<|im_end|>'
                            + '\n<|im_start|>assistant\n'
                        )
                elif role == 'assistant' and isinstance(content, str) and i >= len(self.embedded_messages):
                    content = content + '<|im_end|>\n'

            # LLaMA3 风格
            elif self.model_family == "llama":
                # LLaMA3 常用头：<|start_header_id|>user<|end_header_id|>\n ... <|eot_id|>
                USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n"
                ASSIST_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n"
                EOT = "<|eot_id|>"
                eos_token_id = self.tokenizer.convert_tokens_to_ids(EOT)

                if role == "user" and isinstance(content, str) and i >= len(self.embedded_messages):
                    if i == 0:
                        # 首条 user：注入 plan，引出 <bop>，后面在 embedding 阶段再接 eop + EOT + assistant header
                        content = (
                            USER_HEADER
                            + content
                            + "Now, you are given a step-by-step plan to complete this task as follow: "
                            + "<bop>"
                        )
                        # # plan
                        # content = (
                        #     USER_HEADER
                        #     + content
                        # )
                    else:
                        # 后续 user：自己补上 EOT + assistant header
                        content = (
                            USER_HEADER
                            + content
                            + EOT
                            + ASSIST_HEADER
                        )
                elif role == "assistant" and isinstance(content, str) and i >= len(self.embedded_messages):
                    # assistant 消息后面补 EOT
                    content = content + EOT + "\n"

            # =============== 计算 embedding ===============
            if i >= len(self.embedded_messages):
                embedded_content = get_embedding(
                    content, self.model, self.tokenizer, self.device
                ).to(torch.bfloat16)

                # 对于第一条 user，需要把 hidden_state + 一些 special token embed 拼进去
                if i == 0:
                    if self.model_family == "qwen" or self.model_family == "unknown":
                        # ----- Qwen 特殊 token -----
                        eop_text = '<eop>'
                        eop_input = self.tokenizer(eop_text, return_tensors="pt")
                        eop_input = {k: v.to(self.device) for k, v in eop_input.items()}
                        eop_embed = self.model.get_input_embeddings()(
                            eop_input['input_ids']
                        ).to(torch.bfloat16)

                        end_text = '<|im_end|>\n'
                        end_input = self.tokenizer(end_text, return_tensors="pt")
                        end_input = {k: v.to(self.device) for k, v in end_input.items()}
                        end_embed = self.model.get_input_embeddings()(
                            end_input['input_ids']
                        ).to(torch.bfloat16)

                        assistant_text = '<|im_start|>assistant\n'
                        assistant_input = self.tokenizer(assistant_text, return_tensors="pt")
                        assistant_input = {
                            k: v.to(self.device) for k, v in assistant_input.items()
                        }
                        assistant_embed = self.model.get_input_embeddings()(
                            assistant_input['input_ids']
                        ).to(torch.bfloat16)

                        # hidden_state = hidden_state.to(self.device).to(torch.bfloat16)

                        embedded_content = torch.cat(
                            (
                                embedded_content,
                                hidden_state.unsqueeze(0),
                                eop_embed,
                                end_embed,
                                assistant_embed,
                            ),
                            dim=1,
                        ).to(torch.bfloat16)

                    elif self.model_family == "llama":
                        # ----- LLaMA3 特殊 token -----
                        # 仍然复用 <eop> 表示 plan 结束
                        eop_text = "<eop>"
                        eop_input = self.tokenizer(eop_text, return_tensors="pt")
                        eop_input = {k: v.to(self.device) for k, v in eop_input.items()}
                        eop_embed = self.model.get_input_embeddings()(
                            eop_input["input_ids"]
                        ).to(torch.bfloat16)

                        # LLaMA3 结束一个段落用 <|eot_id|>
                        eot_text = "<|eot_id|>"
                        eot_input = self.tokenizer(eot_text, return_tensors="pt")
                        eot_input = {k: v.to(self.device) for k, v in eot_input.items()}
                        eot_embed = self.model.get_input_embeddings()(
                            eot_input["input_ids"]
                        ).to(torch.bfloat16)

                        # assistant header
                        assistant_text = "<|start_header_id|>assistant<|end_header_id|>\n"
                        assistant_input = self.tokenizer(
                            assistant_text, return_tensors="pt"
                        )
                        assistant_input = {
                            k: v.to(self.device) for k, v in assistant_input.items()
                        }
                        assistant_embed = self.model.get_input_embeddings()(
                            assistant_input["input_ids"]
                        ).to(torch.bfloat16)

                        # hidden_state = hidden_state.to(self.device).to(torch.bfloat16)

                        embedded_content = torch.cat(
                            (
                                embedded_content,
                                hidden_state.unsqueeze(0),
                                eop_embed,
                                eot_embed,
                                assistant_embed,
                            ),
                            dim=1,
                        ).to(torch.bfloat16)

                        # plan
                        # embedded_content = torch.cat(
                        #     (
                        #         embedded_content,
                        #         # hidden_state.unsqueeze(0),
                        #         eop_embed,
                        #         eot_embed,
                        #         assistant_embed,
                        #     ),
                        #     dim=1,
                        # ).to(torch.bfloat16)

                self.embedded_messages.append(
                    {
                        "role": role,
                        "embedding": embedded_content,
                    }
                )

        # 把所有 embedding 拼起来
        embeddings = []
        for msg in self.embedded_messages:
            emb = msg["embedding"]
            if emb.device != self.device:
                emb = emb.to(self.device)
            embeddings.append(emb)

        combined_embedding = torch.cat(embeddings, dim=1)

        # attention mask
        mask_list = []
        for msg in self.embedded_messages:
            emb = msg["embedding"]
            seq_len = emb.size(1)
            mask = torch.ones(seq_len, dtype=torch.long, device=self.device)
            mask_list.append(mask)

        attention_mask = torch.cat(mask_list, dim=0)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": True,
        }

        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=combined_embedding,
                eos_token_id=eos_token_id,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Find the first Action
        action_idx = full_text.find("\nAction:")
        if action_idx != -1:
            next_newline = full_text.find("\n", action_idx + 1)
            if next_newline != -1:
                return full_text[:next_newline]
            else:
                return full_text
        else:
            return full_text