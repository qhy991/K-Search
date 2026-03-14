#!/usr/bin/env python3
"""
PyTorch Profiling Script for Quantized GEMM Kernel
使用 PyTorch 内置的 profiler，不需要 GPU 性能计数器权限
"""

import torch
import torch.profiler as profiler
import json
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "llm_kernel_test"))

def load_kernel_module(attempt_path):
    """加载编译好的 kernel 模块"""
    attempt_dir = project_root / attempt_path
    kernel_cu = attempt_dir / "kernel.cu"
    
    if not kernel_cu.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_cu}")
    
    # 使用 torch.utils.cpp_extension.load 加载
    from torch.utils.cpp_extension import load
    
    module_name = "quant_gemm_kernel"
    module = load(
        name=module_name,
        sources=[str(kernel_cu)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True
    )
    
    return module

def run_profiling(module, M, N, K, device="cuda"):
    """运行 profiling"""
    # 创建测试数据
    # 权重: block_q4_1 格式 (N, K/32)
    num_k_blocks = K // 32
    weight = torch.randint(0, 255, (N, num_k_blocks * 20), dtype=torch.uint8, device=device)
    
    # 激活: FP32 (M, K)
    activation = torch.randn(M, K, dtype=torch.float32, device=device)
    
    # 预热
    for _ in range(5):
        _ = module.forward(weight, activation, M, N, K)
    
    torch.cuda.synchronize()
    
    # 运行 profiling
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with profiler.record_function("quant_gemm_forward"):
            output = module.forward(weight, activation, M, N, K)
            torch.cuda.synchronize()
    
    return prof, output

def analyze_profiling_results(prof, M, N, K):
    """分析 profiling 结果"""
    results = {
        "M": M,
        "N": N,
        "K": K,
        "events": []
    }
    
    # 获取 CUDA 事件
    events = prof.profiler.kineto_results.events()
    
    cuda_events = []
    for event in events:
        if event.device_type() == profiler.ProfilerActivity.CUDA:
            cuda_events.append({
                "name": event.name(),
                "start_us": event.start_us(),
                "end_us": event.end_us(),
                "duration_us": event.duration_us(),
                "memory_usage_bytes": event.cuda_memory_usage() if hasattr(event, 'cuda_memory_usage') else 0,
            })
    
    results["events"] = cuda_events
    
    # 计算总时间
    if cuda_events:
        total_time_us = max(e["end_us"] for e in cuda_events) - min(e["start_us"] for e in cuda_events)
        results["total_time_us"] = total_time_us
        results["total_time_ms"] = total_time_us / 1000.0
        
        # 计算 GFLOPS
        flops = 2 * M * N * K  # GEMM: 2*M*N*K FLOPs
        gflops = (flops / 1e9) / (total_time_us / 1e6)
        results["gflops"] = gflops
    
    return results

def main():
    """主函数"""
    # 配置
    attempt_path = "attempts/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168_profiling"
    output_dir = Path(__file__).parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔍 PyTorch Profiling for Quantized GEMM")
    print(f"Device: {device}")
    print(f"Attempt path: {attempt_path}")
    print()
    
    # 加载 kernel
    print("📦 Loading kernel module...")
    try:
        module = load_kernel_module(attempt_path)
        print("✅ Kernel loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load kernel: {e}")
        return
    
    # 测试配置
    N = 18432
    K = 7168
    test_cases = [
        ("M=1", 1),
        ("M=512", 512),
    ]
    
    all_results = {}
    
    for name, M in test_cases:
        print(f"\n{'='*60}")
        print(f"📊 Profiling {name} (M={M}, N={N}, K={K})")
        print(f"{'='*60}")
        
        try:
            prof, output = run_profiling(module, M, N, K, device)
            results = analyze_profiling_results(prof, M, N, K)
            all_results[name] = results
            
            print(f"✅ Profiling completed")
            print(f"   Total time: {results.get('total_time_ms', 0):.3f} ms")
            print(f"   GFLOPS: {results.get('gflops', 0):.2f}")
            print(f"   CUDA events: {len(results.get('events', []))}")
            
            # 保存详细结果
            output_file = output_dir / f"pytorch_profiling_{name.lower().replace('=', '_')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"   Results saved to: {output_file}")
            
            # 打印关键事件
            if results.get('events'):
                print(f"\n   Key CUDA events:")
                for event in results['events'][:10]:  # 显示前10个事件
                    print(f"     - {event['name']}: {event['duration_us']:.2f} us")
            
        except Exception as e:
            print(f"❌ Profiling failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存汇总结果
    summary_file = output_dir / "pytorch_profiling_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Summary saved to: {summary_file}")
    
    # 打印表格
    print(f"\n{'='*60}")
    print("📊 Performance Summary")
    print(f"{'='*60}")
    print(f"{'Case':<15} {'Time (ms)':<15} {'GFLOPS':<15}")
    print(f"{'-'*45}")
    for name, results in all_results.items():
        time_ms = results.get('total_time_ms', 0)
        gflops = results.get('gflops', 0)
        print(f"{name:<15} {time_ms:<15.3f} {gflops:<15.2f}")

if __name__ == "__main__":
    main()
