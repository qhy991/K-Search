
import torch
import torch.profiler as profiler
import sys
from pathlib import Path

# 添加路径
project_root = Path("/home/qinhaiyan/kernelevalplus")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "llm_kernel_test"))

from llm_kernel_test.test_runner import KernelTestRunner
import json

# 加载定义和编译 kernel
runner = KernelTestRunner()
definition_path = project_root / "definitions/quant_gemm/deepseek_v3/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168.json"
attempt_path = project_root / "attempts/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168_profiling"

with open(definition_path) as f:
    definition = json.load(f)

runner.definition_path = str(definition_path)
runner.batch_dir = str(attempt_path.parent)

attempt_id = attempt_path.name
spec = runner.definition_converter.convert_definition_to_spec(definition)

# 编译
compile_result = runner.compiler.compile_kernel(
    attempt_id=attempt_id,
    spec=spec,
    batch_dir=str(attempt_path.parent)
)

if not compile_result["success"]:
    print(f"Compilation failed: {compile_result.get('errors', [])}")
    sys.exit(1)

module = compile_result["module"]

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
        activities=[profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        output = module.forward(weight, activation, M, N, K)
        torch.cuda.synchronize()
    
    # 分析结果
    events = prof.profiler.kineto_results.events()
    cuda_events = [e for e in events if e.device_type() == profiler.ProfilerActivity.CUDA]
    
    if cuda_events:
        total_time = sum(e.duration_us() for e in cuda_events) / 1000.0
        flops = 2 * M * N * K
        gflops = (flops / 1e9) / (total_time / 1000.0)
        
        results[name] = {
            "time_ms": total_time,
            "gflops": gflops,
            "num_events": len(cuda_events)
        }
        
        print(f"   ✅ Time: {total_time:.3f} ms")
        print(f"   ✅ GFLOPS: {gflops:.2f}")
        print(f"   ✅ CUDA events: {len(cuda_events)}")
        
        # 保存 trace
        trace_file = f"/home/qinhaiyan/kernelevalplus/output/glm-5/quant_gemm/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168/pytorch_trace_{name.lower().replace('=', '_')}.json"
        prof.export_chrome_trace(trace_file)
        print(f"   ✅ Chrome trace: {trace_file}")
    
    # 打印表格
    print(f"\n{'='*60}")
    print("📊 Performance Summary")
    print(f"{'='*60}")
    print(f"{'Case':<15} {'Time (ms)':<15} {'GFLOPS':<15}")
    print(f"{'-'*45}")
    for name, res in results.items():
        print(f"{name:<15} {res['time_ms']:<15.3f} {res['gflops']:<15.2f}")

# 保存汇总
summary_file = "/home/qinhaiyan/kernelevalplus/output/glm-5/quant_gemm/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168/pytorch_profiling_summary.json"
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Summary saved: {summary_file}")
