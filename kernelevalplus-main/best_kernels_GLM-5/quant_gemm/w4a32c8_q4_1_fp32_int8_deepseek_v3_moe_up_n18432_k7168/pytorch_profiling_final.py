#!/usr/bin/env python3
"""
PyTorch Profiling - 最终版本
直接使用 torch.profiler 对已编译的 kernel 进行 profiling
"""

import torch
import torch.profiler as profiler
import json
import sys
import importlib.util
from pathlib import Path

def load_compiled_module(attempt_dir):
    """加载已编译的模块"""
    attempt_path = Path(attempt_dir)
    
    # 查找 .so 文件
    so_files = list(attempt_path.glob("*.so"))
    if not so_files:
        raise FileNotFoundError(f"No .so file found in {attempt_path}")
    
    so_file = so_files[0]
    module_name = so_file.stem
    
    # 使用 importlib 加载
    spec = importlib.util.spec_from_file_location(module_name, so_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for {so_file}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module

def main():
    project_root = Path("/home/qinhaiyan/kernelevalplus")
    attempt_dir = project_root / "attempts/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168_profiling"
    output_dir = Path(__file__).parent
    
    print("🔍 PyTorch Profiling for Quantized GEMM")
    print("=" * 60)
    print()
    
    # 加载模块
    print("📦 Loading compiled kernel module...")
    try:
        module = load_compiled_module(attempt_dir)
        print("✅ Module loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load module: {e}")
        print("\n💡 Trying to compile first...")
        
        # 如果加载失败，先运行一次测试来编译
        import subprocess
        result = subprocess.run([
            "python", "llm_kernel_test/test_runner.py", "--test",
            "--definition", "definitions/quant_gemm/deepseek_v3/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168.json",
            "--attempt-path", "attempts/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168_profiling",
            "--variant", "W4A32C8"
        ], cwd=project_root, capture_output=True)
        
        if result.returncode != 0:
            print("❌ Compilation failed")
            return
        
        # 再次尝试加载
        try:
            module = load_compiled_module(attempt_dir)
            print("✅ Module loaded after compilation")
        except Exception as e2:
            print(f"❌ Still failed to load: {e2}")
            return
    
    # 测试配置
    N = 18432
    K = 7168
    test_cases = [("M=1", 1), ("M=512", 512)]
    
    results = {}
    
    for name, M in test_cases:
        print(f"\n{'='*60}")
        print(f"📊 Profiling {name}")
        print(f"{'='*60}")
        
        try:
            # 创建测试数据
            num_k_blocks = K // 32
            weight = torch.randint(0, 255, (N, num_k_blocks * 20), dtype=torch.uint8, device="cuda")
            activation = torch.randn(M, K, dtype=torch.float32, device="cuda")
            
            # 预热
            print("   Warming up...")
            for _ in range(5):
                _ = module.forward(weight, activation, M, N, K)
            torch.cuda.synchronize()
            
            # Profiling
            print("   Running profiling...")
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
            
            # 分析结果
            events = prof.profiler.kineto_results.events()
            cuda_events = [e for e in events if e.device_type() == profiler.ProfilerActivity.CUDA]
            
            if cuda_events:
                # 计算总时间（所有 CUDA 事件）
                total_time_us = sum(e.duration_us() for e in cuda_events)
                total_time_ms = total_time_us / 1000.0
                
                # 计算 GFLOPS
                flops = 2 * M * N * K
                gflops = (flops / 1e9) / (total_time_ms / 1000.0)
                
                results[name] = {
                    "time_ms": total_time_ms,
                    "time_us": total_time_us,
                    "gflops": gflops,
                    "num_events": len(cuda_events),
                    "events": [
                        {
                            "name": e.name(),
                            "duration_us": e.duration_us(),
                            "start_us": e.start_us(),
                        }
                        for e in cuda_events[:20]  # 保存前20个事件
                    ]
                }
                
                print(f"   ✅ Total time: {total_time_ms:.3f} ms ({total_time_us:.2f} us)")
                print(f"   ✅ GFLOPS: {gflops:.2f}")
                print(f"   ✅ CUDA events: {len(cuda_events)}")
                
                # 保存 Chrome trace
                trace_file = output_dir / f"pytorch_trace_{name.lower().replace('=', '_')}.json"
                prof.export_chrome_trace(str(trace_file))
                print(f"   ✅ Chrome trace saved: {trace_file}")
                
                # 打印关键事件
                print(f"\n   Top 10 CUDA events by duration:")
                sorted_events = sorted(cuda_events, key=lambda e: e.duration_us(), reverse=True)
                for i, event in enumerate(sorted_events[:10], 1):
                    print(f"     {i:2d}. {event.name():<50} {event.duration_us():>10.2f} us")
                
            else:
                print("   ⚠️  No CUDA events found")
                
        except Exception as e:
            print(f"   ❌ Profiling failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存汇总
    if results:
        summary_file = output_dir / "pytorch_profiling_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Summary saved: {summary_file}")
        
        # 打印表格
        print(f"\n{'='*60}")
        print("📊 Performance Summary")
        print(f"{'='*60}")
        print(f"{'Case':<15} {'Time (ms)':<15} {'GFLOPS':<15} {'Events':<10}")
        print(f"{'-'*55}")
        for name, res in results.items():
            print(f"{name:<15} {res['time_ms']:<15.3f} {res['gflops']:<15.2f} {res['num_events']:<10}")
    
    print(f"\n✅ Profiling completed!")
    print(f"\n💡 提示: 可以在 Chrome 浏览器中打开 chrome://tracing 查看详细的 trace 文件")

if __name__ == "__main__":
    main()
