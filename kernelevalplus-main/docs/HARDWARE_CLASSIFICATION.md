# KernelEvalPlus Benchmark - Hardware Classification

This document describes the hardware classification used in the benchmark system.

## Hardware Categories

### 1. Laptop GPUs
- **RTX 4070 Laptop** (Mobile)
- **RTX 5070 Laptop** (Mobile)

**Baseline Reference**: RTX 4070

### 2. Desktop GPUs
- **RTX 4090** (Consumer Desktop)

**Baseline Reference**: RTX 4090

### 3. Server/Data Center GPUs
- **H800** (Data Center)
- **A100** (Data Center)
- **A800** (Data Center)

**Baseline Reference**: H800

## Baseline Priority Mapping

When comparing performance, the system automatically selects the appropriate baseline hardware based on the local GPU type:

```python
BASELINE_PRIORITY = {
    "laptop": "RTX4070",      # For RTX 4070/5070 Laptop
    "desktop": "RTX4090",     # For RTX 4090 Desktop
    "server": "H800",         # For H800/A100/A800 Server
}
```

## Detection Keywords

The system detects hardware type using these keywords:

```python
HARDWARE_TYPE_KEYWORDS = {
    "laptop": ["laptop", "mobile", "notebook", "4070", "5070"],
    "desktop": ["4090", "rtx", "gtx", "geforce"],
    "server": ["a100", "h100", "h800", "a40", "a30", "a10", "l40", "v100"],
}
```

## Baseline Data

All baseline performance data is stored in:
- `core/tools/baseline_data_compact.json`

Each entry contains performance metrics for multiple hardware configurations:

```json
{
  "w4a32c8_q4_0_f32_m4096_n1_k4096": {
    "hardware": {
      "RTX4090": { "tflops": 4.95, "gflops": 4950.0 },
      "RTX4070": { "tflops": 1.39, "gflops": 1390.0 },  // Laptop
      "RTX5070": { "tflops": 0.186, "gflops": 186.25 }, // Laptop
      "H800": { "tflops": 3.8, "gflops": 3800.0 },
      "A100": { "tflops": 2.54, "gflops": 2540.0 }
    }
  }
}
```

## Environment Override

You can override hardware detection by setting the `KEVAL_HARDWARE` environment variable:

```bash
export KEVAL_HARDWARE="RTX4070"  # Force laptop mode
export KEVAL_HARDWARE="RTX4090"  # Force desktop mode
export KEVAL_HARDWARE="H800"     # Force server mode
```

## Performance Comparison

When viewing results in the WebUI, the system will:

1. Detect local GPU (or use `KEVAL_HARDWARE`)
2. Determine hardware category (laptop/desktop/server)
3. Select appropriate baseline for comparison
4. Display performance ratio vs baseline

Example:
- Running on **RTX 5070 Laptop** → compares with **RTX 4070 baseline**
- Running on **RTX 4090 Desktop** → compares with **RTX 4090 baseline**
- Running on **H800 Server** → compares with **H800 baseline**
