# Kernels Tab 集成方案

## 1. 数据文件

- ✅ `data/kernels_summary.json` - 已生成，包含 138 个 kernels

## 2. HTML 结构更新

在现有 tab 按钮后添加 "Kernels" tab:

```html
<button class="tab" data-tab="kernels">Kernels</button>
```

## 3. 页面设计（参考 FlashInfer Bench）

### 3.1 按算子类型视图

```
┌─────────────────────────────────────────────────────────┐
│ Kernels                                                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 算子类型: [Quant GEMM ▼]                               │
│                                                         │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Quant GEMM (95 kernels)                            │ │
│ │                                                     │ │
│ │ DeepSeek-V2    18 kernels                          │ │
│ │ DeepSeek-V3    26 kernels                          │ │
│ │ LLaMA          31 kernels                          │ │
│ │ ...                                                 │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Kernel 详情:                                            │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ w4a32c8_q4_0_fp32_int8_deepseek_v2_att_out_n5120_k5120 │ │
│ │ Variant: W4A32C8                                    │ │
│ │ Model: DeepSeek-V2                                  │ │
│ │ Axes: M(var), N=5120, K=5120                        │ │
│ │ Test Configs: batch_1, batch_2, ...                │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 3.2 按模型视图

```
┌─────────────────────────────────────────────────────────┐
│ Models                                                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ LLaMA (31 kernels)                                     │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Quant GEMM     9 kernels                           │ │
│ │ Flash Attention 11 kernels                         │ │
│ │ RMS Norm        5 kernels                           │ │
│ │ TopK            6 kernels                           │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 4. 算子类型卡片样式

参考 FlashInfer 设计，每个算子类型显示为卡片：

```css
.kernel-type-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
}

.kernel-type-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.kernel-type-icon {
    width: 40px;
    height: 40px;
    background: #f0f0f0;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.kernel-count {
    background: #e3f2fd;
    color: #1976d2;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 0.9em;
}
```

## 5. JavaScript 功能

### 5.1 加载数据

```javascript
let kernelsData = null;

async function loadKernelsData() {
    const response = await fetch(BASE + 'data/kernels_summary.json');
    kernelsData = await response.json();
    console.log('Loaded', kernelsData.total_kernels, 'kernels');
}
```

### 5.2 渲染函数

```javascript
function renderKernelsByOperator() {
    // 按算子类型显示 kernels
}

function renderKernelsByModel() {
    // 按模型显示 kernels
}

function renderKernelDetails(kernelInfo) {
    // 显示 kernel 详情
}
```

## 6. 实现步骤

1. ✅ 生成 kernels_summary.json
2. 添加 "Kernels" tab 到 HTML
3. 添加 CSS 样式
4. 添加 JavaScript 加载和渲染逻辑
5. 测试和调试

## 7. 数据映射

算子类型 -> 显示名称:
- quant_gemm -> Quant GEMM
- flash_attention -> Flash Attention
- rms_norm -> RMS Norm
- topk -> TopK Sampling

模型名称映射:
- deepseek_v2 -> DeepSeek-V2
- llama -> LLaMA
- llama3_8b -> LLaMA-3-8B
- qwen -> Qwen
- mixtral8x7b -> Mixtral-8x7B
