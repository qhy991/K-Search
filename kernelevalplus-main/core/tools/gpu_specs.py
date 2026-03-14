
"""
GPU Specifications and Definitions for Prompt Generation
"""

# Hardware type (from config) to GPU Model name mapping
# This ensures consistency between hardware.type config and GPU specs
HARDWARE_TYPE_TO_GPU_MODEL = {
    # Lowercase hardware type -> GPU Model name in GPU_SPEC_INFO
    "a800": "A800-80G",
    "a100": "A100-80GB",
    "a100-40g": "A100",
    "h100": "H100",
    "4090": "RTX 4090",
    "5070": "RTX 5070",
    "l40s": "L40S",
    "l4": "L4",
    "t4": "T4",
    "a10g": "A10G",
    # Legacy/alternative names
    "rtx4090": "RTX 4090",
    "rtx5070": "RTX 5070",
    "ada": "L40S",  # Generic Ada architecture -> L40S
}

def get_gpu_model_from_hardware_type(hardware_type: str) -> str:
    """
    Convert hardware type (from config) to GPU model name (for GPU_SPEC_INFO).

    Args:
        hardware_type: Hardware type string from config (e.g., "a800", "4090", "A800", "A100-80GB")

    Returns:
        GPU model name for GPU_SPEC_INFO (e.g., "A800-80G", "RTX 4090")
        Falls back to "A100-80GB" if hardware_type is not recognized.

    Example:
        >>> get_gpu_model_from_hardware_type("a800")
        "A800-80G"
        >>> get_gpu_model_from_hardware_type("A800")  # Case insensitive
        "A800-80G"
        >>> get_gpu_model_from_hardware_type("4090")
        "RTX 4090"
        >>> get_gpu_model_from_hardware_type("A100-80GB")  # Direct model name
        "A100-80GB"
    """
    if not hardware_type:
        return "A100-80GB"  # Default fallback

    # First, check if it's already a valid GPU model name (for direct input)
    if hardware_type in GPU_SPEC_INFO:
        return hardware_type

    # Normalize to lowercase for case-insensitive matching
    hardware_type_lower = hardware_type.lower().strip()

    # Lookup in mapping table
    gpu_model = HARDWARE_TYPE_TO_GPU_MODEL.get(hardware_type_lower)

    if gpu_model:
        return gpu_model
    else:
        # Fallback: try to match GPU model names case-insensitively
        for model_name in GPU_SPEC_INFO.keys():
            if model_name.lower() == hardware_type_lower or hardware_type_lower in model_name.lower():
                return model_name

        # Final fallback
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Unknown hardware type '{hardware_type}', falling back to A100-80GB")
        return "A100-80GB"


def detect_gpu_hardware() -> str:
    """
    Auto-detect the GPU hardware from the system.

    Returns:
        GPU model name (e.g., "RTX 4090", "H100", "A100-80GB")
        Falls back to "A100-80GB" if detection fails.

    Detection methods (tried in order):
    1. nvidia-smi command line tool
    2. PyTorch CUDA detection
    3. Default fallback
    """
    import subprocess
    import re

    # Method 1: Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip().split("\n")[0].strip()
            # Parse GPU name and map to our model names
            return _map_detected_gpu_to_model(gpu_name)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 2: Try PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return _map_detected_gpu_to_model(gpu_name)
    except ImportError:
        pass

    # Fallback
    return "A100-80GB"


def _map_detected_gpu_to_model(detected_name: str) -> str:
    """
    Map detected GPU name to our GPU model names.

    Args:
        detected_name: Raw GPU name from nvidia-smi or PyTorch

    Returns:
        Our GPU model name (e.g., "RTX 4090", "H100")
    """
    detected_lower = detected_name.lower()

    # Check for exact matches first
    for model_name in GPU_SPEC_INFO.keys():
        if model_name.lower() in detected_lower:
            return model_name

    # Check for partial matches with priority order
    # Hopper H100
    if "h100" in detected_lower or "hopper" in detected_lower:
        return "H100"

    # Ada Lovelace - RTX 4090, 4080, 5070, L40S, L4
    if "rtx 4090" in detected_lower or "geforce rtx 4090" in detected_lower:
        return "RTX 4090"
    if "rtx 4080" in detected_lower:
        return "RTX 4080"
    if "rtx 5070" in detected_lower or "geforce rtx 5070" in detected_lower:
        return "RTX 5070"
    if "l40s" in detected_lower:
        return "L40S"
    if "l4" in detected_lower and "l40" not in detected_lower:
        return "L4"

    # Ampere - A100, A800
    if "a100" in detected_lower:
        # Check if it's 80GB or 40GB
        if "80" in detected_lower or "80gb" in detected_lower:
            return "A100-80GB"
        return "A100"
    if "a800" in detected_lower or "a800-80g" in detected_lower:
        return "A800-80G"
    if "a10g" in detected_lower:
        return "A10G"
    if "a30" in detected_lower:
        return "A30"

    # Turing - T4
    if "t4" in detected_lower:
        return "T4"
    if "rtx 2080" in detected_lower:
        return "RTX 2080"

    # Volta - V100
    if "v100" in detected_lower:
        return "V100"

    # Blackwell - B100, B200, RTX 5090, RTX 5080
    if "b100" in detected_lower:
        return "B100"
    if "b200" in detected_lower:
        return "B200"
    if "rtx 5090" in detected_lower:
        return "RTX 5090"
    if "rtx 5080" in detected_lower:
        return "RTX 5080"

    # Default fallback
    return "A100-80GB"

GPU_SPEC_INFO = {
    "L40S": {
        "GPU Architecture": "Ada",
        "GPU Memory": "48GB GDDR6 with ECC",
        "Memory Bandwidth": "864 GB/s",
        "RT Core Performance TFLOPS": "212",
        "FP32 TFLOPS": "91.6",
        "TF32 Tensor Core TFLOPS": "183.2 (366 with sparsity)",
        "FP16 Tensor Core TFLOPS": "362.05 (733 with sparsity)",
        "FP8 Tensor Core TFLOPS": "733 (1466 with sparsity)",
        "Peak INT8 Tensor TOPS": "733 (1466 with sparsity)",
        "Peak INT4 Tensor TOPS": "733 (1466 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "24",
        "Shared memory capacity per SM": "100 KB",
        "Maximum shared memory per thread block": "99 KB",
    },
    "H100": {
        "GPU Architecture": "Hopper",
        "GPU Memory": "80GB",
        "Memory Bandwidth": "3.35 TB/s",
        "FP64 TFLOPS": "34",
        "FP64 Tensor Core TFLOPS": "67",
        "FP32 TFLOPS": "67",
        "TF32 Tensor Core TFLOPS": "989 with sparsity",
        "BFLOAT16 Tensore Core TFLOPS": "1979 with sparsity",
        "FP16 Tensor Core TFLOPS": "1979 with sparsity",
        "FP8 Tensor Core TFLOPS": "3958 with sparsity",
        "INT8 Tensor Core TOPS": "3958 with sparsity",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "228 KB",
        "Maximum shared memory per thread block": "227 KB",
    },
    # this is 40GB (Standard)
    "A100": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "40GB",
        "Memory Bandwidth": "1935 GB/s",
        "FP64 TFLOPS": "9.7",
        "FP64 Tensor Core TFLOPS": "19.5",
        "FP32 TFLOPS": "19.5",
        "TF32 Tensor Core TFLOPS": "156 (312 with sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "312 (624 with sparsity)",
        "FP16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "INT8 Tensor Core TOPS": "624 (1248 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "A100-80GB": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "80GB",
        "Memory Bandwidth": "1935 GB/s",
        "FP64 TFLOPS": "9.7",
        "FP64 Tensor Core TFLOPS": "19.5",
        "FP32 TFLOPS": "19.5",
        "TF32 Tensor Core TFLOPS": "156 (312 with sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "312 (624 with sparsity)",
        "FP16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "INT8 Tensor Core TOPS": "624 (1248 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "L4": {
        "GPU Architecture": "Ada",
        "GPU Memory": "24GB",
        "Memory Bandwidth": "300 GB/s",
        "FP32 TFLOPS": "30.3",
        "TF32 Tensor Core TFLOPS": "120 with sparsity",
        "BFLOAT16 Tensore Core TFLOPS": "242 with sparsity",
        "FP8 Tensor Core TFLOPS": "485 with sparsity",
        "INT8 Tensor Core TOPS": "485 with sparsity",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "24",
        "Shared memory capacity per SM": "100 KB",
        "Maximum shared memory per thread block": "99 KB",
    },
    "T4": {
        "GPU Architecture": "Turing",
        "GPU Memory": "16 GB GDDR6",
        "Memory Bandwidth": "300 GB/s",
        "Single-Precision TFLOPS": "8.1",
        "Mixed-Precision (FP16/FP32) TFLOPS": "65",
        "INT8 TOPS": "130",
        "INT4 TOPS": "260",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "16",
        "Shared memory capacity per SM": "64 KB",
    },
    "A10G": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "24GB GDDR6",
        "Memory Bandwidth": "600 GB/s",
        "FP32 TFLOPS": "31.2",
        "TF32 Tensor Core TFLOPS": "62.5 (125 with sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "125 (250 with sparsity)",
        "FP16 Tensor Core TFLOPS": "125 (250 with sparsity)",
        "INT8 Tensor Core TOPS": "250 (500 with sparsity)",
        "INT4 Tensor Core TOPS": "500 (1000 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "A800-80G": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "80GB",
        "Memory Bandwidth": "1935 GB/s",
        "FP64 TFLOPS": "9.7",
        "FP64 Tensor Core TFLOPS": "19.5",
        "FP32 TFLOPS": "19.5",
        "TF32 Tensor Core TFLOPS": "156 (312 with sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "312 (624 with sparsity)",
        "FP16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "INT8 Tensor Core TOPS": "624 (1248 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "RTX 4090": {
        "GPU Architecture": "Ada Lovelace",
        "GPU Memory": "24GB GDDR6X",
        "Memory Bandwidth": "1008 GB/s",
        "FP32 TFLOPS": "80.6",
        "RT Core Performance TFLOPS": "191.4",
        "Tensor Core Performance": "642.9 TOPS (FP16), 1285.8 TOPS (INT8)",
        "CUDA Cores": "16384",
        "RT Cores": "128",
        "Tensor Cores": "512",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "100 KB",
        "Maximum shared memory per thread block": "99 KB",
    },
    "RTX 5070": {
        "GPU Architecture": "Blackwell",
        "GPU Memory": "16GB GDDR7",
        "Memory Bandwidth": "896 GB/s",
        "FP32 TFLOPS": "~60",
        "RT Core Performance TFLOPS": "~142",
        "Tensor Core Performance": "~475 TOPS (FP16), ~950 TOPS (INT8)",
        "CUDA Cores": "12288",
        "RT Cores": "96",
        "Tensor Cores": "384",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "128 KB",
        "Maximum shared memory per thread block": "127 KB",
    }
}

# Basic GPU concept definitions
GPU_DEFINITIONS = {
    "Thread": "A thread is a single execution unit that can run a single instruction at a time.",
    "Thread Block": "A thread block is a group of threads that can cooperate with each other.",
    "Warp": "A warp is a group of threads that are scheduled together and execute in parallel.",
    "Shared Memory": "Shared memory is a memory space that can be accessed by all threads in a thread block.",
    "Register": "A register is a small memory space that can be accessed by a single thread.",
    "Memory Hierarchy": "Memory hierarchy is a pyramid of memory types with different speeds and sizes.",
    "Memory Bandwidth": "Memory bandwidth is the rate at which data can be read from or stored into memory.",
    "Cache": "Cache is a small memory space that stores frequently accessed data.",
    "HBM": "HBM is a high-bandwidth memory technology that uses 3D-stacked DRAM.",
}

GPU_BEST_PRACTICES = [
    # From https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
    # CUDA Best Practices Section
    "Find ways to parallelize sequential code.",
    "Minimize data transfers between the host and the device.",
    "Adjust kernel launch configuration to maximize device utilization.",
    "Ensure that global memory accesses are coalesced.",
    "Minimize redundant accesses to global memory whenever possible.",
    "Avoid long sequences of diverged execution by threads within the same warp.",
    # we added this to reference the specific GPU architecture
    "Use specialized instructions based on the specific GPU architecture",
]

def get_gpu_specs_str(gpu_model: str = "A800-80G") -> str:
    """
    Format GPU specs for a specific model into a string for the prompt.
    """
    specs = GPU_SPEC_INFO.get(gpu_model)
    if not specs:
        # Fallback for unknown models or generic request
        # Return generic advice or maybe A100 as a reference?
        # For now, just return empty string or a warning
        return f"Warning: No specifications found for GPU model '{gpu_model}'. Using generic CUDA optimization strategies."

    spec_str = f"### Hardware Specifications ({gpu_model})\n"
    for key, value in specs.items():
        spec_str += f"- **{key}**: {value}\n"

    return spec_str

def get_gpu_definitions_str() -> str:
    """
    Format GPU definitions into a string.
    """
    def_str = "### Core GPU Concepts\n"
    for key, value in GPU_DEFINITIONS.items():
        def_str += f"- **{key}**: {value}\n"
    return def_str

def get_gpu_best_practices_str() -> str:
    """
    Format best practices into a string.
    """
    bp_str = "### Optimization Best Practices\n"
    for bp in GPU_BEST_PRACTICES:
        bp_str += f"- {bp}\n"
    return bp_str


# ============================================================================
# Architecture-Specific Optimization Knowledge Base
# ============================================================================

# GPU Model to Architecture mapping
GPU_MODEL_TO_ARCHITECTURE = {
    "H100": "Hopper",
    "H200": "Hopper",
    "L40S": "Ada",
    "L4": "Ada",
    "RTX 4090": "Ada",
    "RTX 4080": "Ada",
    "A100": "Ampere",
    "A100-80GB": "Ampere",
    "A800-80G": "Ampere",
    "A10G": "Ampere",
    "A30": "Ampere",
    "T4": "Turing",
    "RTX 2080": "Turing",
    "V100": "Volta",
    "RTX 5070": "Blackwell",
    "RTX 5080": "Blackwell",
    "RTX 5090": "Blackwell",
    "B100": "Blackwell",
    "B200": "Blackwell",
}

# Architecture-specific optimization techniques
ARCHITECTURE_OPTIMIZATIONS = {
    "Hopper": {
        "name": "Hopper (SM 9.0)",
        "compute_capability": "9.0",
        "key_features": [
            "Thread Block Clusters for SM-to-SM communication",
            "TMA (Tensor Memory Accelerator) for async bulk data movement",
            "Native FP8 support with Transformer Engine",
            "WGMMA (Warpgroup Matrix Multiply-Accumulate) instructions",
            "Distributed Shared Memory across clusters",
            "50MB L2 Cache",
        ],
        "memory_bound": [
            {
                "technique": "TMA Async Bulk Copy",
                "description": "Use Tensor Memory Accelerator for efficient bulk async transfers between global and shared memory",
                "code_hint": "Use cuda::memcpy_async with TMA descriptors, or CUTLASS TMA copy operations",
                "expected_improvement": "30-50% memory throughput improvement",
                "cuda_version": "12.0+",
            },
            {
                "technique": "Thread Block Clusters",
                "description": "Leverage distributed shared memory across SM clusters to reduce global memory traffic",
                "code_hint": "__cluster_dims__(X, Y, Z) launch attribute; use cluster.sync() for synchronization",
                "expected_improvement": "Reduce global memory access by sharing data across SMs",
                "cuda_version": "12.0+",
            },
            {
                "technique": "Large L2 Cache Optimization",
                "description": "With 50MB L2 cache, more aggressive data caching strategies are viable",
                "code_hint": "Consider L2 persistence hints: cudaAccessPolicyWindow for frequently accessed data",
                "expected_improvement": "Better cache hit rates for working sets up to 50MB",
            },
        ],
        "compute_bound": [
            {
                "technique": "WGMMA Instructions",
                "description": "Use Warpgroup Matrix Multiply-Accumulate for higher Tensor Core utilization",
                "code_hint": "Use CUTLASS 3.x with Hopper-specific kernels; wgmma.mma_async instructions",
                "expected_improvement": "Higher sustained TFLOPS than wmma",
            },
            {
                "technique": "FP8 Tensor Cores",
                "description": "Native FP8 (E4M3/E5M2) support for inference workloads",
                "code_hint": "Use __nv_fp8_e4m3 / __nv_fp8_e5m2 types; CUTLASS FP8 GEMM kernels",
                "expected_improvement": "2x throughput vs FP16 for suitable workloads",
            },
        ],
        "occupancy_limited": [
            {
                "technique": "Cluster-level Occupancy",
                "description": "Tune cluster size to balance occupancy and shared memory usage",
                "code_hint": "Experiment with cluster sizes: 1x1, 2x1, 2x2; check cudaOccupancyMaxPotentialClusterSize",
                "expected_improvement": "Better resource utilization with optimal cluster config",
            },
        ],
    },
    "Ada": {
        "name": "Ada Lovelace (SM 8.9)",
        "compute_capability": "8.9",
        "key_features": [
            "4th Generation Tensor Cores with FP8 support",
            "Shader Execution Reordering (SER)",
            "100KB shared memory per SM",
            "Improved L2 cache efficiency",
            "Native FP8 (E4M3, E5M2) data types",
        ],
        "memory_bound": [
            {
                "technique": "cp.async with Large Shared Memory",
                "description": "Use async copy to fully utilize 100KB shared memory for larger tiles",
                "code_hint": "asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\"); use 100KB for tiling",
                "expected_improvement": "Larger tiles reduce global memory transactions",
            },
            {
                "technique": "L2 Cache Persistence",
                "description": "Use L2 cache access policy for frequently accessed data",
                "code_hint": "cudaStreamSetAttribute with cudaAccessPolicyWindow",
                "expected_improvement": "Reduced DRAM traffic for reused data",
            },
            {
                "technique": "Vectorized float4 Loads",
                "description": "128-bit vectorized loads for coalesced access patterns",
                "code_hint": "float4* ptr = reinterpret_cast<float4*>(data); ensure 16-byte alignment",
                "expected_improvement": "4x reduction in memory transactions",
            },
        ],
        "compute_bound": [
            {
                "technique": "FP8 Tensor Cores",
                "description": "Native FP8 support for inference with 2x throughput vs FP16",
                "code_hint": "__nv_fp8_e4m3 for weights, __nv_fp8_e5m2 for gradients",
                "expected_improvement": "2x Tensor Core throughput for suitable precision",
            },
            {
                "technique": "TF32 Tensor Cores",
                "description": "Use TF32 for FP32-like precision with Tensor Core acceleration",
                "code_hint": "cublasTensorOpMath policy; wmma with tf32 fragments",
                "expected_improvement": "8x speedup over FP32 CUDA cores",
            },
        ],
        "occupancy_limited": [
            {
                "technique": "Tune Block Size for 100KB Shared Memory",
                "description": "Adjust block size to balance shared memory and occupancy",
                "code_hint": "With 100KB/SM, can use larger blocks; check cudaOccupancyMaxPotentialBlockSize",
                "expected_improvement": "Find optimal occupancy-tile size tradeoff",
            },
        ],
    },
    "Ampere": {
        "name": "Ampere (SM 8.0/8.6)",
        "compute_capability": "8.0/8.6",
        "key_features": [
            "3rd Generation Tensor Cores",
            "Async Copy (cp.async) for shared memory prefetch",
            "TF32 for FP32 simulation with Tensor Cores",
            "Fine-grained structured sparsity (2:4)",
            "164KB shared memory per SM (A100) / 100KB (A10)",
            "L2 cache residency control",
        ],
        "memory_bound": [
            {
                "technique": "cp.async + Double Buffering",
                "description": "Use async copy to overlap data loading with computation",
                "code_hint": "asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\"); cp.async.commit_group; cp.async.wait_group<N>;",
                "expected_improvement": "20-40% latency hiding through overlapped loads",
            },
            {
                "technique": "L2 Residency Control",
                "description": "Pin frequently accessed data in L2 cache",
                "code_hint": "cudaStreamSetAttribute with cudaStreamAttrValue.accessPolicyWindow",
                "expected_improvement": "Improved cache hit rate for hot data",
            },
            {
                "technique": "Shared Memory Tiling",
                "description": "Use 164KB shared memory (A100) for larger tiles",
                "code_hint": "__shared__ float tile[TILE_M][TILE_K]; experiment with tile sizes up to 164KB",
                "expected_improvement": "Better data reuse, fewer global memory accesses",
            },
        ],
        "compute_bound": [
            {
                "technique": "TF32 Tensor Cores",
                "description": "Use TF32 for FP32 precision needs with 8x Tensor Core speedup",
                "code_hint": "cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH); wmma::mma_sync with tf32",
                "expected_improvement": "8x theoretical speedup over FP32 CUDA cores",
            },
            {
                "technique": "Structured Sparsity (2:4)",
                "description": "2:4 fine-grained sparsity for 2x Tensor Core throughput",
                "code_hint": "Use cusparseLt library; requires weight pruning to 2:4 pattern",
                "expected_improvement": "2x Tensor Core throughput with sparse weights",
            },
            {
                "technique": "Mixed Precision (FP16/BF16)",
                "description": "Use FP16/BF16 accumulation for memory and compute efficiency",
                "code_hint": "__half2 for vectorized FP16; __nv_bfloat16 for BF16",
                "expected_improvement": "2x memory bandwidth, higher Tensor Core utilization",
            },
        ],
        "occupancy_limited": [
            {
                "technique": "Tune Shared Memory vs L1 Cache",
                "description": "Adjust shared memory / L1 split with cudaFuncSetAttribute",
                "code_hint": "cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, ratio)",
                "expected_improvement": "Optimize for kernel's shared memory needs",
            },
            {
                "technique": "Register Pressure Reduction",
                "description": "Reduce register usage to increase occupancy",
                "code_hint": "__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM); -maxrregcount=N",
                "expected_improvement": "Higher occupancy for latency-sensitive kernels",
            },
        ],
    },
    "Turing": {
        "name": "Turing (SM 7.5)",
        "compute_capability": "7.5",
        "key_features": [
            "2nd Generation Tensor Cores",
            "Independent Thread Scheduling",
            "Unified cache architecture",
            "64KB shared memory per SM",
            "INT8/INT4 Tensor Core support",
        ],
        "memory_bound": [
            {
                "technique": "Vectorized Loads (float4)",
                "description": "Use 128-bit loads for better memory coalescing",
                "code_hint": "float4* ptr = reinterpret_cast<float4*>(data); load 4 floats at once",
                "expected_improvement": "4x reduction in memory transactions",
            },
            {
                "technique": "Shared Memory Optimization",
                "description": "Use 64KB shared memory efficiently with proper bank conflict avoidance",
                "code_hint": "__shared__ float smem[TILE][TILE+1]; // +1 to avoid bank conflicts",
                "expected_improvement": "Reduced bank conflicts, better shared memory throughput",
            },
            {
                "technique": "__ldg() Intrinsic",
                "description": "Use texture cache path for read-only data",
                "code_hint": "float val = __ldg(&read_only_data[idx]);",
                "expected_improvement": "Better cache utilization for read-only access",
            },
        ],
        "compute_bound": [
            {
                "technique": "INT8 Tensor Cores",
                "description": "Use INT8 for quantized inference with high throughput",
                "code_hint": "wmma::mma_sync with int8 fragments; CUTLASS int8 GEMM",
                "expected_improvement": "4x throughput vs FP16 for quantized models",
            },
            {
                "technique": "FP16 Tensor Cores",
                "description": "Use FP16 accumulation for compute-intensive workloads",
                "code_hint": "wmma::mma_sync<16,16,16>; half fragments",
                "expected_improvement": "8x throughput over FP32 CUDA cores",
            },
        ],
        "occupancy_limited": [
            {
                "technique": "Block Size Tuning",
                "description": "Find optimal block size for 64KB shared memory limit",
                "code_hint": "Try 128, 256, 512 threads; check occupancy with cudaOccupancyMaxPotentialBlockSize",
                "expected_improvement": "Balance between parallelism and resource usage",
            },
        ],
    },
    "Volta": {
        "name": "Volta (SM 7.0)",
        "compute_capability": "7.0",
        "key_features": [
            "1st Generation Tensor Cores",
            "Independent Thread Scheduling",
            "Combined L1/Shared Memory (128KB)",
            "HBM2 memory",
        ],
        "memory_bound": [
            {
                "technique": "Configurable L1/Shared Split",
                "description": "Adjust the 128KB L1/shared memory split based on kernel needs",
                "code_hint": "cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared)",
                "expected_improvement": "Optimal cache configuration for workload",
            },
            {
                "technique": "Coalesced Memory Access",
                "description": "Ensure threads in a warp access consecutive memory addresses",
                "code_hint": "int idx = blockIdx.x * blockDim.x + threadIdx.x; access data[idx]",
                "expected_improvement": "Maximize HBM2 bandwidth utilization",
            },
        ],
        "compute_bound": [
            {
                "technique": "FP16 Tensor Cores (WMMA)",
                "description": "Use wmma API for Tensor Core matrix operations",
                "code_hint": "wmma::fragment, wmma::load_matrix_sync, wmma::mma_sync",
                "expected_improvement": "8x throughput vs FP32 for supported shapes",
            },
        ],
        "occupancy_limited": [
            {
                "technique": "Warp-Level Primitives",
                "description": "Use warp shuffle for efficient intra-warp communication",
                "code_hint": "__shfl_down_sync(0xffffffff, val, delta); reduces shared memory need",
                "expected_improvement": "Faster reductions, less shared memory pressure",
            },
        ],
    },
    "Blackwell": {
        "name": "Blackwell (SM 10.0)",
        "compute_capability": "10.0",
        "key_features": [
            "5th Generation Tensor Cores",
            "FP4 support for extreme quantization",
            "Enhanced TMA with new features",
            "Larger shared memory and L2 cache",
            "Improved memory bandwidth (GDDR7/HBM3e)",
        ],
        "memory_bound": [
            {
                "technique": "Enhanced TMA Operations",
                "description": "Next-generation Tensor Memory Accelerator with improved async capabilities",
                "code_hint": "Use CUTLASS 3.x+ with Blackwell-optimized kernels",
                "expected_improvement": "Further improved async memory throughput",
            },
            {
                "technique": "GDDR7/HBM3e Optimization",
                "description": "Leverage higher memory bandwidth with optimized access patterns",
                "code_hint": "Maximize coalescing and vectorization for new memory subsystem",
                "expected_improvement": "Better utilization of increased bandwidth",
            },
        ],
        "compute_bound": [
            {
                "technique": "FP4 Tensor Cores",
                "description": "Ultra-low precision for extreme throughput in inference",
                "code_hint": "Emerging support; check latest CUDA/CUTLASS for FP4 APIs",
                "expected_improvement": "4x throughput vs FP8 for suitable workloads",
            },
            {
                "technique": "5th Gen Tensor Cores",
                "description": "Improved Tensor Core architecture with higher throughput",
                "code_hint": "Use latest CUTLASS/cuBLAS for Blackwell-optimized paths",
                "expected_improvement": "Highest Tensor Core performance in NVIDIA lineup",
            },
        ],
        "occupancy_limited": [
            {
                "technique": "Larger Resource Limits",
                "description": "Blackwell supports larger shared memory and register files",
                "code_hint": "Experiment with larger tiles and more registers per thread",
                "expected_improvement": "More flexibility in kernel design",
            },
        ],
    },
}


def get_gpu_architecture(gpu_model: str) -> str:
    """
    Get the architecture name for a given GPU model.

    Args:
        gpu_model: GPU model name (e.g., "A800-80G", "H100", "RTX 4090")

    Returns:
        Architecture name (e.g., "Ampere", "Hopper", "Ada")
    """
    return GPU_MODEL_TO_ARCHITECTURE.get(gpu_model, "Unknown")


def get_architecture_from_hardware_type(hardware_type: str) -> str:
    """
    Get architecture name from hardware type config.

    Args:
        hardware_type: Hardware type from config (e.g., "a800", "h100")

    Returns:
        Architecture name
    """
    gpu_model = get_gpu_model_from_hardware_type(hardware_type)
    return get_gpu_architecture(gpu_model)


def get_architecture_optimizations(architecture: str, bottleneck_type: str = None) -> dict:
    """
    Get architecture-specific optimization recommendations.

    Args:
        architecture: GPU architecture name (e.g., "Ampere", "Hopper")
        bottleneck_type: Optional bottleneck type to filter optimizations
                        ("memory_bound", "compute_bound", "occupancy_limited")

    Returns:
        Dictionary with architecture info and relevant optimizations
    """
    arch_info = ARCHITECTURE_OPTIMIZATIONS.get(architecture, {})

    if not arch_info:
        return {
            "architecture": architecture,
            "warning": f"No optimization data available for architecture '{architecture}'",
            "optimizations": [],
        }

    result = {
        "architecture": architecture,
        "name": arch_info.get("name", architecture),
        "compute_capability": arch_info.get("compute_capability", "Unknown"),
        "key_features": arch_info.get("key_features", []),
    }

    if bottleneck_type:
        # Return optimizations for specific bottleneck
        bottleneck_key = bottleneck_type.lower().replace(" ", "_").replace("-", "_")
        # Handle variations: "Memory Bound" -> "memory_bound"
        if "memory" in bottleneck_key:
            bottleneck_key = "memory_bound"
        elif "compute" in bottleneck_key:
            bottleneck_key = "compute_bound"
        elif "occupancy" in bottleneck_key or "latency" in bottleneck_key:
            bottleneck_key = "occupancy_limited"

        result["bottleneck_type"] = bottleneck_type
        result["optimizations"] = arch_info.get(bottleneck_key, [])
    else:
        # Return all optimizations
        result["memory_bound"] = arch_info.get("memory_bound", [])
        result["compute_bound"] = arch_info.get("compute_bound", [])
        result["occupancy_limited"] = arch_info.get("occupancy_limited", [])

    return result


def format_architecture_optimizations_str(
    architecture: str,
    bottleneck_type: str = None,
    max_techniques: int = 3
) -> str:
    """
    Format architecture-specific optimizations as a markdown string for prompts.

    Args:
        architecture: GPU architecture name
        bottleneck_type: Optional bottleneck type to focus on
        max_techniques: Maximum number of techniques to include per category

    Returns:
        Formatted markdown string
    """
    arch_data = get_architecture_optimizations(architecture, bottleneck_type)

    if "warning" in arch_data:
        return f"⚠️ {arch_data['warning']}\n"

    lines = []
    lines.append(f"### Architecture-Specific Optimizations ({arch_data['name']})")
    lines.append(f"**Compute Capability**: SM {arch_data['compute_capability']}\n")

    # Key features
    lines.append("**Key Architecture Features**:")
    for feature in arch_data.get("key_features", [])[:5]:
        lines.append(f"- {feature}")
    lines.append("")

    # Optimizations
    if bottleneck_type and "optimizations" in arch_data:
        lines.append(f"**Recommended Optimizations for {bottleneck_type}**:")
        for opt in arch_data["optimizations"][:max_techniques]:
            lines.append(f"\n**{opt['technique']}**")
            lines.append(f"- Description: {opt['description']}")
            lines.append(f"- Code Hint: `{opt['code_hint']}`")
            lines.append(f"- Expected Impact: {opt['expected_improvement']}")
            if "cuda_version" in opt:
                lines.append(f"- Requires: CUDA {opt['cuda_version']}")
    else:
        # Show top optimization from each category
        for category in ["memory_bound", "compute_bound", "occupancy_limited"]:
            opts = arch_data.get(category, [])
            if opts:
                category_name = category.replace("_", " ").title()
                lines.append(f"\n**{category_name} Optimizations**:")
                for opt in opts[:2]:  # Top 2 per category
                    lines.append(f"- **{opt['technique']}**: {opt['description']}")

    return "\n".join(lines)
