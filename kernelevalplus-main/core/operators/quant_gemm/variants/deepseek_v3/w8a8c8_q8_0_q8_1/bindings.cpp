#include <torch/extension.h>
#include <cuda_runtime.h>

// External C wrapper function declaration
extern "C" void gemm_w8a8c8_q8_0_q8_1(
    const uint8_t* weight,
    const float* activation,
    float* output,
    int M,
    int N,
    int K
);

torch::Tensor gemm_w8a8c8_q8_0_q8_1_wrapper(
    torch::Tensor weight,
    torch::Tensor activation,
    int M,
    int N,
    int K
) {
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "Activation must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "Weight must be uint8");
    TORCH_CHECK(activation.dtype() == torch::kFloat32, "Activation must be float32");
    TORCH_CHECK(K % 32 == 0, "K must be divisible by 32");

    // Create output tensor
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(weight.device()));

    // Get data pointers
    const uint8_t* weight_ptr = weight.data_ptr<uint8_t>();
    const float* activation_ptr = activation.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Call C wrapper function
    gemm_w8a8c8_q8_0_q8_1(weight_ptr, activation_ptr, output_ptr, M, N, K);

    return output;
}

PYBIND11_MODULE(w8a8c8_q8_0_q8_1_binding, m) {
    m.def("gemm_w8a8c8_q8_0_q8_1", &gemm_w8a8c8_q8_0_q8_1_wrapper,
        "Quantized GEMM with Q8_0 weights and Q8_1-style activations (llama.cpp pattern, DeepSeek-V3 W8A8C8)",
        py::arg("weight"), py::arg("activation"),
        py::arg("M"), py::arg("N"), py::arg("K"));
}
