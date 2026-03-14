#!/usr/bin/env python3
"""
修改使用 closest match 的定义文件，使其 N, K 维度与 baseline 匹配
"""

import json
import re
from pathlib import Path

# 需要修改的定义文件及其新的维度
MODIFICATIONS = [
    # DeepSeek-V3 att_qkv: N=21504→18432, K=7168
    {
        'model_dir': 'deepseek_v3',
        'old_file': 'w8a32c8_q8_0_fp32_int8_ds3_att_qkv_n21504_k7168.json',
        'new_file': 'w8a32c8_q8_0_fp32_int8_ds3_att_qkv_n18432_k7168.json',
        'old_n': 21504, 'old_k': 7168,
        'new_n': 18432, 'new_k': 7168,
    },
    # LLaMA-3-70B att_out: N=8192→7168, K=8192→7168
    {
        'model_dir': 'llama',
        'old_file': 'w8a32c8_q8_0_fp32_int8_llama3_70b_att_out_n8192_k8192.json',
        'new_file': 'w8a32c8_q8_0_fp32_int8_llama3_70b_att_out_n7168_k7168.json',
        'old_n': 8192, 'old_k': 8192,
        'new_n': 7168, 'new_k': 7168,
    },
    # LLaMA-3-8B att_qkv: N=12288, K=4096→5120
    {
        'model_dir': 'llama',
        'old_file': 'w8a32c8_q8_0_fp32_int8_llama3_8b_att_qkv_n12288_k4096.json',
        'new_file': 'w8a32c8_q8_0_fp32_int8_llama3_8b_att_qkv_n12288_k5120.json',
        'old_n': 12288, 'old_k': 4096,
        'new_n': 12288, 'new_k': 5120,
    },
    # Qwen2.5-7B att_qkv: N=10752→9728, K=3584→2560
    {
        'model_dir': 'qwen2_5_7b',
        'old_file': 'w8a32c8_q8_0_fp32_int8_qwen2_5_7b_att_qkv_n10752_k3584.json',
        'new_file': 'w8a32c8_q8_0_fp32_int8_qwen2_5_7b_att_qkv_n9728_k2560.json',
        'old_n': 10752, 'old_k': 3584,
        'new_n': 9728, 'new_k': 2560,
    },
    # Qwen3-0.6B att_qkv: N=3072→5120, K=1024→1536
    {
        'model_dir': 'qwen3',
        'old_file': 'w8a32c8_q8_0_fp32_int8_qwen3_0_6b_att_qkv_n3072_k1024.json',
        'new_file': 'w8a32c8_q8_0_fp32_int8_qwen3_0_6b_att_qkv_n5120_k1536.json',
        'old_n': 3072, 'old_k': 1024,
        'new_n': 5120, 'new_k': 1536,
    },
    # Qwen3-1.5B att_qkv: N=7680→7168, K=2560→2048
    {
        'model_dir': 'qwen3',
        'old_file': 'w8a32c8_q8_0_fp32_int8_qwen3_1_5b_att_qkv_n7680_k2560.json',
        'new_file': 'w8a32c8_q8_0_fp32_int8_qwen3_1_5b_att_qkv_n7168_k2048.json',
        'old_n': 7680, 'old_k': 2560,
        'new_n': 7168, 'new_k': 2048,
    },
    # Qwen3-14B ffn_up: N=17408→18944, K=5120→3584
    {
        'model_dir': 'qwen3',
        'old_file': 'w8a32c8_q8_0_fp32_int8_qwen3_14b_ffn_up_n17408_k5120.json',
        'new_file': 'w8a32c8_q8_0_fp32_int8_qwen3_14b_ffn_up_n18944_k3584.json',
        'old_n': 17408, 'old_k': 5120,
        'new_n': 18944, 'new_k': 3584,
    },
    # Qwen3-32B ffn_up: N=25600→18944, K=5120→3584
    {
        'model_dir': 'qwen3',
        'old_file': 'w8a32c8_q8_0_fp32_int8_qwen3_32b_ffn_up_n25600_k5120.json',
        'new_file': 'w8a32c8_q8_0_fp32_int8_qwen3_32b_ffn_up_n18944_k3584.json',
        'old_n': 25600, 'old_k': 5120,
        'new_n': 18944, 'new_k': 3584,
    },
    # Qwen3-8B att_qkv: N=12288, K=4096→5120
    {
        'model_dir': 'qwen3',
        'old_file': 'w8a32c8_q8_0_fp32_int8_qwen3_8b_att_qkv_n12288_k4096.json',
        'new_file': 'w8a32c8_q8_0_fp32_int8_qwen3_8b_att_qkv_n12288_k5120.json',
        'old_n': 12288, 'old_k': 4096,
        'new_n': 12288, 'new_k': 5120,
    },
]


def modify_definition_file(mod: dict, definitions_dir: Path):
    """修改定义文件的 N, K 维度"""
    old_file_path = definitions_dir / mod['model_dir'] / mod['old_file']

    if not old_file_path.exists():
        print(f"⚠️  文件不存在: {old_file_path}")
        return False

    # 读取原文件
    with open(old_file_path) as f:
        data = json.load(f)

    # 修改 N, K 值
    old_n = mod['old_n']
    old_k = mod['old_k']
    new_n = mod['new_n']
    new_k = mod['new_k']

    # 更新 axes
    data['axes']['N']['value'] = new_n
    data['axes']['K']['value'] = new_k

    # 更新 name (如果包含维度)
    old_name = data['name']
    new_name = old_name.replace(f'_n{old_n}_k{old_k}', f'_n{new_n}_k{new_k}')
    data['name'] = new_name

    # 更新 description 中的维度（如果有）
    desc = data.get('description', '')
    desc = re.sub(r'\bN=\d+', f'N={new_n}', desc)
    desc = re.sub(r'\bK=\d+', f'K={new_k}', desc)
    data['description'] = desc

    # 更新 test_configs 中的 N, K 值
    for config in data.get('test_configs', []):
        if config.get('N') == old_n:
            config['N'] = new_n
        if config.get('K') == old_k:
            config['K'] = new_k

    # 更新 formula 中的维度
    formula = data.get('formula', {})
    for key, value in formula.items():
        if isinstance(value, str):
            value = re.sub(rf'\b{old_n}\b', str(new_n), value)
            value = re.sub(rf'\b{old_k}\b', str(new_k), value)
            formula[key] = value

    # 保存新文件
    new_file_path = definitions_dir / mod['model_dir'] / mod['new_file']
    with open(new_file_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ {mod['old_file']} -> {mod['new_file']}")
    print(f"   N: {old_n} -> {new_n}, K: {old_k} -> {new_k}")

    # 删除旧文件
    old_file_path.unlink()

    return True


def main():
    definitions_dir = Path('/home/haiyan/Agent4Kernel/KernelEvalPlus/definitions/quant_gemm')

    print('=' * 100)
    print('修改使用 closest match 的定义文件，使其 N, K 维度与 baseline 匹配')
    print('=' * 100)
    print()

    success_count = 0
    fail_count = 0

    for mod in MODIFICATIONS:
        try:
            if modify_definition_file(mod, definitions_dir):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"❌ 修改失败: {mod['old_file']} - {e}")
            fail_count += 1

    print()
    print('📊 总结:')
    print(f'  成功修改: {success_count} 个')
    print(f'  修改失败: {fail_count} 个')
    print(f'  总计: {len(MODIFICATIONS)} 个')


if __name__ == '__main__':
    main()
