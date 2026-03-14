#!/usr/bin/env python3
"""
简化版 PyTorch Profiling - 直接使用测试框架
"""

import torch
import torch.profiler as profiler
import json
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def profile_with_torch_profiler():
    """使用 PyTorch Profiler 进行 profiling"""
    
    # 导入测试框架
    from llm_kernel_test.test_runner import load_definition, test_kernel
    
    # 加载定义
    definition_path = project_root / "definitions/quant_gemm/deepseek_v3/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168.json"
    attempt_path = project_root / "attempts/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168_profiling"
    output_dir = Path(__file__).parent
    
    print("🔍 PyTorch Profiling for Quantized GEMM")
    print(f"Definition: {definition_path}")
    print(f"Attempt: {attempt_path}")
    print()
    
    # 加载定义
    with open(definition_path) as f:
        spec = json.load(f)
    
    # 加载 kernel（通过测试框架）
    # 这里我们需要直接调用 kernel，而不是通过测试框架
    # 所以我们需要手动加载模块
    
    from torch.utils.cpp_extension import load
    
    # 读取 kernel 代码
    kernel_file = attempt_path / "kernel.cu"
    
    if not kernel_file.exists():
        print(f"❌ Kernel file not found: {kernel_file}")
        return {}
    
    # 加载模块
    print("📦 Loading kernel...")
    try:
        module = load(
            name="quant_gemm_profiling",
            sources=[str(kernel_file)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False
        )
        print("✅ Kernel loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load kernel: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    # 测试配置
    N = 18432
    K = 7168
    test_cases = [("M=1", 1), ("M=512", 512)]
    
    results = {}
    
    for name, M in test_cases:
        print(f"\n{'='*60}")
        print(f"📊 Profiling {name}")
        print(f"{'='*60}")
        
        # 创建测试数据
        num_k_blocks = K // 32
        weight = torch.randint(0, 255, (N, num_k_blocks * 20), dtype=torch.uint8, device="cuda")
        activation = torch.randn(M, K, dtype=torch.float32, device="cuda")
        
        # 预热
        for _ in range(3):
            _ = module.forward(weight, activation, M, N, K)
        torch.cuda.synchronize()
        
        # Profiling
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with profiler.record_function("quant_gemm"):
                output = module.forward(weight, activation, M, N, K)
                torch.cuda.synchronize()
        
        # 分析结果
        events = prof.profiler.kineto_results.events()
        cuda_events = [e for e in events if e.device_type() == profiler.ProfilerActivity.CUDA]
        
        if cuda_events:
            total_time = sum(e.duration_us() for e in cuda_events) / 1000.0  # ms
            flops = 2 * M * N * K
            gflops = (flops / 1e9) / (total_time / 1000.0)
            
            results[name] = {
                "time_ms": total_time,
                "gflops": gflops,
                "num_events": len(cuda_events)
            }
            
            print(f"✅ Time: {total_time:.3f} ms")
            print(f"✅ GFLOPS: {gflops:.2f}")
            print(f"✅ CUDA events: {len(cuda_events)}")
            
            # 保存详细 trace
            trace_file = output_dir / f"pytorch_trace_{name.lower().replace('=', '_')}.json"
            prof.export_chrome_trace(str(trace_file))
            print(f"✅ Chrome trace saved: {trace_file}")
    
    # 保存汇总
    summary_file = output_dir / "pytorch_profiling_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Summary saved: {summary_file}")
    
    return results

if __name__ == "__main__":
    from torch.utils.cpp_extension import load
    profile_with_torch_profiler()
