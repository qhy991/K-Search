# Performance Comparison Chart

## TFLOPS Comparison by Version

```
Version | M=1    | M=8    | M=512  | Notes
--------|--------|--------|--------|-------
v1      | 0.427  | 0.499  | 0.390  | Baseline
v2      | 0.449  | 0.483  | 0.385  | ILP attempt
v4      | 0.426  | 0.499  | 0.319  | Simplified
v7      | 0.554  | 0.495  | 25.928 | DP4A + Tensor Core
v8      | 1.645  | 0.497  | 25.758 | FINAL (Best for M=1)
--------|--------|--------|--------|-------
Best    | 1.645  | 0.497  | 25.928 | v8 for M=1, v7 for M=512
Ref     | 1.631  | 0.501  | 26.976 | GGML reference
```

## Latency Comparison (ms)

```
Version | M=1    | M=8    | M=512
--------|--------|--------|--------
v1      | 2.454  | 16.799 | 1375.049
v2      | 2.337  | 17.383 | 1394.988
v4      | 2.464  | 16.802 | 1682.251
v7      | 1.894  | 16.948 | 20.706
v8      | 0.637  | 16.875 | 20.843
--------|--------|--------|--------
Ref     | 0.643  | 16.750 | 19.902
```

## Key Improvements

| Version | M=1 Improvement | M=512 Improvement |
|---------|----------------|-------------------|
| v1→v7  | 1.30× (0.427→0.554) | 66× (0.39→25.928) |
| v7→v8  | 2.97× (0.554→1.645) | -0.7% (25.928→25.758) |
| v1→v8  | **3.85×** | **66×** |

## Correctness (NMSE)

All versions passed the NMSE threshold of 0.05:

| Version | NMSE    | Status |
|---------|---------|--------|
| v1      | 0.000000 | ✅ |
| v2      | 0.000000 | ✅ |
| v4      | 0.000000 | ✅ |
| v7      | 0.000027 | ✅ |
| v8      | 0.000029 | ✅ |

---

## Best Version Selection

**For production use, use v8** - it achieves the best overall performance:
- **M=1**: 1.645 TFLOPS (exceeds reference by 0.9%)
- **M=8**: 0.497 TFLOPS (99.2% of reference)
- **M=512**: 25.758 TFLOPS (95.5% of reference)
- **Correctness**: NMSE = 0.000029 (excellent)
