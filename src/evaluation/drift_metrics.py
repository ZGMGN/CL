"""Numerical metrics for sequential LoRA diagnostics (no model loading)."""
import math

import torch
from peft import PeftConfig, load_peft_weights

#计算cosine距离
def cosine(first, second):
    first, second = first.detach().double().reshape(-1), second.detach().double().reshape(-1)
    #detach()用于脱离当前计算图
    if first.shape != second.shape:
        raise ValueError('Cosine inputs must have equal shapes')
    norm = first.norm() * second.norm()
    #norm是求当前L2范数，即各个元素平方和开根
    return None if norm == 0 else float((first @ second / norm).clamp(-1, 1))
    #clamp(min,max)把数压缩到[min,max],这里已经归一化还用clamp原因是计算机计算浮点数可能会为1.000..02


def representation_cosine_drift(first, second):
    first, second = first.double(), second.double()
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError('Representations must be matching [examples, features] matrices')
    norms = first.norm(dim=1) * second.norm(dim=1)
    valid = norms > 0
    if not valid.any():
        return None
    return float((1 - ((first * second).sum(1)[valid] / norms[valid]).clamp(-1, 1)).mean())
#1.这里不flatten，而是按照dim=1计算，是逐样本计算再加权，否则可能被极端样本干扰
#2.这里用1-cosine，结果属于[0,2]，0代表重合，1代表正交，2代表相反

def linear_cka(first, second):
#线性cka分析
    first, second = first.double(), second.double()
    if first.ndim != 2 or second.ndim != 2 or len(first) != len(second) or len(first) < 2:
        raise ValueError('CKA requires at least two aligned examples')
    #cka要求样本数必须相同
    grams = []
    for values in (first, second):
        values = values - values.mean(0, keepdim=True)
        gram = values @ values.T
        grams.append(gram - gram.mean(0, keepdim=True) - gram.mean(1, keepdim=True) + gram.mean())
    #一套中心化小操作
    norm = grams[0].norm() * grams[1].norm()
    if norm == 0:
        return None
    return float(((grams[0] * grams[1]).sum() / norm).clamp(0, 1))
    #cka的公式为 XX^T@YY^T.sum()/XX^T_L2*YY^T_L2

#散度
def distribution_drift(log_before, log_after):
    #这里输入的已经是log了
    """Mean next-token KL in nats over the entire vocabulary, on identical prompts."""
    if log_before.shape != log_after.shape:
        raise ValueError('Distributions must have matching shapes')
    forward, reverse = [], []
    for before, after in zip(log_before, log_after):
        before, after = before.double(), after.double()
        forward.append(float((before.exp() * (before - after)).sum().clamp_min(0)))
        #exp是逐元素还原
        reverse.append(float((after.exp() * (after - before)).sum().clamp_min(0)))
    return {'kl_forward': sum(forward) / len(forward),
            'kl_reverse': sum(reverse) / len(reverse),
            'symmetric_kl': (sum(forward) + sum(reverse)) / (2 * len(forward))}


def load_lora_factors(folder):
    """Load effective standard-LoRA BA factors, including alpha/r scaling."""
    config = PeftConfig.from_pretrained(str(folder))
    if (config.peft_type != 'LORA' or config.use_dora or config.modules_to_save
            or config.bias != 'none' or getattr(config, 'lora_bias', False)):
        raise ValueError('LoRA drift supports standard bias-free LoRA without DoRA/modules_to_save')
    state = load_peft_weights(str(folder), device='cpu')
    factors = {}
    for key, value in state.items():
        if not key.endswith('.lora_A.weight'):
            continue
        module = key[:-len('.lora_A.weight')]
        bkey = module + '.lora_B.weight'
        if bkey not in state:
            raise ValueError(f'Missing LoRA B for {module}')
        # The project's training configuration uses a uniform rank and alpha.
        if config.rank_pattern or config.alpha_pattern:
            raise ValueError('Nonuniform rank/alpha patterns are not supported by this diagnostic')
            #这里rank_pattern和alpha_pattern指的是某些层专门用的pattern，而一般我们都是指定全局pattern
        rank = config.r
        scale = config.lora_alpha / (math.sqrt(rank) if config.use_rslora else rank)
        factors[module] = (state[bkey].float(), value.float(), scale)
    if not factors:
        raise ValueError(f'No standard linear LoRA factors found in {folder}')
    expected_keys = {module + suffix for module in factors for suffix in ('.lora_A.weight', '.lora_B.weight')}
    if set(state) != expected_keys:
        raise ValueError('Adapter contains unsupported parameters beyond linear LoRA A/B weights')
    return factors


def factor_inner(first, second):
    """Frobenius <s1 B1 A1, s2 B2 A2>, without materializing dense updates."""
    b1, a1, scale1 = first
    b2, a2, scale2 = second
    if b1.shape[0] != b2.shape[0] or a1.shape[1] != a2.shape[1]:
        raise ValueError('LoRA module shapes do not match')
    return float(((b1.double().T @ b2.double()) * (a1.double() @ a2.double().T)).sum()) * scale1 * scale2


def update_inner(after1, before1, after2, before2):
    """Inner product of two incremental updates; empty factors mean the base model."""
    modules = set(after1) | set(before1) | set(after2) | set(before2)
    #键取交集，去重
    result = 0.0
    for module in modules:
        for left, sign1 in ((after1, 1), (before1, -1)):
            for right, sign2 in ((after2, 1), (before2, -1)):
                if module in left and module in right:
                    result += sign1 * sign2 * factor_inner(left[module], right[module])
    return result
#u = w_after - w_before 等价于直接矩阵减再计算

def lora_update_cosine(after1, before1, after2, before2):
    dot = update_inner(after1, before1, after2, before2)
    square1 = max(0.0, update_inner(after1, before1, after1, before1))
    square2 = max(0.0, update_inner(after2, before2, after2, before2))
    if square1 == 0 or square2 == 0:
        return None
    return max(-1.0, min(1.0, dot / math.sqrt(square1 * square2)))
#计算<u1,u2>/sqrt(<u1,u1>*<u2,u2>)
