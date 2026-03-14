#!/usr/bin/env python3
"""
最简单的 PyTorch Profiling - 直接运行测试并捕获结果
"""

import subprocess
import sys
from pathlib import Path

def main():
    project_root = Path("/home/qinhaiyan/kernelevalplus")
    output_dir = Path(__file__).parent
    
    print("🔍 PyTorch Profiling for Quantized GEMM")
    print("=" * 60)
    print()
    print("💡 提示: 这个方法会运行测试并显示性能数据")
    print("   如果需要详细的 profiling，建议:")
    print("   1. 使用 Chrome trace 查看时间线")
    print("   2. 或者解决 NCU 权限问题")
    print()
    
    # 运行测试
    print("🚀 Running test with performance data...")
    print()
    
    result = subprocess.run([
        "python", "llm_kernel_test/test_runner.py", "--test",
        "--definition", "definitions/quant_gemm/deepseek_v3/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168.json",
        "--attempt-path", "attempts/w4a32c8_q4_1_fp32_int8_deepseek_v3_moe_up_n18432_k7168_profiling",
        "--variant", "W4A32C8"
    ], cwd=project_root, text=True)
    
    print()
    print("=" * 60)
    
    if result.returncode == 0:
        print("✅ Test completed successfully!")
        print()
        print("📊 性能数据已在上面的输出中显示")
        print()
        print("💡 如果需要更详细的 profiling:")
        print("   - 查看 test_results.json 文件获取详细数据")
        print("   - 或者解决 NCU 权限问题以获取 GPU 指标")
    else:
        print("❌ Test failed")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
