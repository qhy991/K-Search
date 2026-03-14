#!/usr/bin/env python3
"""
直接使用 PyTorch Profiler - 通过测试框架加载 kernel
"""

import torch
import torch.profiler as profiler
import json
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "llm_kernel_test"))

def main():
    """主函数"""
    # 配置
    attempt_path = project_root / "attempts/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168_profiling"
    definition_path = project_root / "definitions/quant_gemm/deepseek_v3/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168.json"
    output_dir = Path(__file__).parent
    
    print("🔍 PyTorch Profiling for Quantized GEMM")
    print(f"Definition: {definition_path}")
    print(f"Attempt: {attempt_path}")
    print()
    
    # 使用测试框架加载和编译 kernel
    print("📦 Loading kernel using test framework...")
    try:
        from llm_kernel_test.test_runner import KernelTestRunner
        
        runner = KernelTestRunner()
        runner.definition_path = str(definition_path)
        runner.batch_dir = str(attempt_path.parent)
        
        # 加载定义
        with open(definition_path) as f:
            definition = json.load(f)
        
        # 编译 kernel
        attempt_id = attempt_path.name
        spec = runner.definition_converter.convert_definition_to_spec(definition)
        
        # 编译
        compile_result = runner.compiler.compile_kernel(
            attempt_id=attempt_id,
            spec=spec,
            batch_dir=str(attempt_path.parent)
        )
        
        if not compile_result["success"]:
            print(f"❌ Compilation failed: {compile_result.get('errors', [])}")
            return
        
        print("✅ Kernel compiled successfully")
        
        # 加载模块
        module = compile_result["module"]
        
    except Exception as e:
        print(f"❌ Failed to load kernel: {e}")
        import traceback
        traceback.print_exc()
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
            for _ in range(3):
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
                with profiler.record_function("quant_gemm"):
                    output = module.forward(weight, activation, M, N, K)
                    torch.cuda.synchronize()
            
            # 分析结果
            events = prof.profiler.kineto_results.events()
            cuda_events = [e for e in events if e.device_type() == profiler.ProfilerActivity.CUDA]
            
            if cuda_events:
                # 计算总时间
                kernel_events = [e for e in cuda_events if "gemm" in e.name().lower() or "kernel" in e.name().lower()]
                if kernel_events:
                    total_time = sum(e.duration_us() for e in kernel_events) / 1000.0  # ms
                else:
                    total_time = sum(e.duration_us() for e in cuda_events) / 1000.0  # ms
                
                flops = 2 * M * N * K
                gflops = (flops / 1e9) / (total_time / 1000.0)
                
                results[name] = {
                    "time_ms": total_time,
                    "gflops": gflops,
                    "num_events": len(cuda_events),
                    "kernel_events": len(kernel_events)
                }
                
                print(f"   ✅ Time: {total_time:.3f} ms")
                print(f"   ✅ GFLOPS: {gflops:.2f}")
                print(f"   ✅ CUDA events: {len(cuda_events)}")
                print(f"   ✅ Kernel events: {len(kernel_events)}")
                
                # 保存详细 trace
                trace_file = output_dir / f"pytorch_trace_{name.lower().replace('=', '_')}.json"
                prof.export_chrome_trace(str(trace_file))
                print(f"   ✅ Chrome trace saved: {trace_file}")
                
                # 打印关键事件
                print(f"\n   Key CUDA events:")
                for event in cuda_events[:10]:
                    print(f"     - {event.name()}: {event.duration_us():.2f} us")
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
        print(f"{'Case':<15} {'Time (ms)':<15} {'GFLOPS':<15}")
        print(f"{'-'*45}")
        for name, res in results.items():
            print(f"{name:<15} {res['time_ms']:<15.3f} {res['gflops']:<15.2f}")
    
    print(f"\n✅ Profiling completed!")

if __name__ == "__main__":
    main()
