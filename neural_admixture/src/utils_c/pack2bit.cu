#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm> // Para std::min

#define THREADS_PER_BLOCK 256
#define MAX_ROWS_PER_BATCH 1024

__global__ void pack2bit_kernel(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                 int batch_rows, int M, int packed_cols) {
    // Calcular índices globales
    int row = blockIdx.y;
    int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Salir si estamos fuera de los límites del batch
    if (row >= batch_rows || chunk_idx >= packed_cols)
        return;

    int row_offset_in = row * M;
    int row_offset_out = row * packed_cols;
    int in_col = chunk_idx * 4;  // Cada hilo procesa 4 valores

    // Empaquetamiento
    uint8_t packed = 0;
    for (int i = 0; i < 4; ++i) {
        int idx = in_col + i;
        if (idx < M) { // Asegurar que no excedemos las columnas originales M
            uint8_t val = input[row_offset_in + idx] & 0x03; // Tomar solo los 2 bits menos significativos
            packed |= (val << (i * 2));
        }
    }

    // Escribir resultado
    output[row_offset_out + chunk_idx] = packed;
}

__global__ void unpack2bit_kernel(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                  int batch_rows, int M, int packed_cols) {
    // Calcular índices globales
    int row = blockIdx.y;
    int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Salir si estamos fuera de los límites del batch
    if (row >= batch_rows || chunk_idx >= packed_cols)
        return;

    int row_offset_in = row * packed_cols;
    int row_offset_out = row * M;
    int out_col = chunk_idx * 4;  // Cada valor empaquetado contiene 4 valores

    // Obtener valor empaquetado
    uint8_t packed = input[row_offset_in + chunk_idx];

    // Desempaquetar (4 valores por byte)
    for (int i = 0; i < 4; ++i) {
        int idx = out_col + i;
        if (idx < M) { // Asegurar que no excedemos las columnas originales M
            output[row_offset_out + idx] = (packed >> (i * 2)) & 0x03; // Extraer los 2 bits correspondientes
        }
    }
}

// Función para comprimir datos desde CPU y dejar el resultado en GPU
void pack2bit_cpu_to_gpu_cuda(torch::Tensor input_cpu, torch::Tensor output_gpu) {
    // Verificar que los tensores estén en los dispositivos correctos
    TORCH_CHECK(input_cpu.device().is_cpu(), "Input tensor must be on CPU");
    TORCH_CHECK(output_gpu.device().is_cuda(), "Output tensor must be on CUDA device");

    int N = input_cpu.size(0);
    int M = input_cpu.size(1);
    int packed_cols = (M + 3) / 4;

    // Verificar que las dimensiones coinciden
    TORCH_CHECK(output_gpu.size(0) == N, "Output tensor row dimension mismatch");
    TORCH_CHECK(output_gpu.size(1) == packed_cols, "Output tensor column dimension mismatch");

    // Determinar el tamaño del batch
    int rows_per_batch = std::min(MAX_ROWS_PER_BATCH, N);

    // Crear tensor temporal para batch de entrada en GPU
    auto options_input = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto input_gpu_batch_temp = torch::empty({rows_per_batch, M}, options_input);

    // Configuración de grid y bloques
    dim3 threads(THREADS_PER_BLOCK);

    // Procesar por lotes
    for (int start_row = 0; start_row < N; start_row += rows_per_batch) {
        int current_batch_rows = std::min(rows_per_batch, N - start_row);

        // Crear vistas para el batch actual en CPU y GPU
        auto input_cpu_batch_view = input_cpu.slice(0, start_row, start_row + current_batch_rows);

        // Redimensionar la vista del tensor GPU temporal si este batch es más pequeño que el máximo
        auto input_gpu_batch_view = (current_batch_rows < rows_per_batch) ?
                                    input_gpu_batch_temp.slice(0, 0, current_batch_rows) : input_gpu_batch_temp;

        // Vista del tensor de salida final en GPU para este batch
        auto output_gpu_batch_view = output_gpu.slice(0, start_row, start_row + current_batch_rows);

        // Copiar batch de CPU a GPU temporal
        input_gpu_batch_view.copy_(input_cpu_batch_view);

        // Configurar grid para este batch
        dim3 blocks((packed_cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, current_batch_rows);

        // Lanzar kernel usando los datos del tensor temporal en GPU como entrada
        pack2bit_kernel<<<blocks, threads>>>(
            input_gpu_batch_view.data_ptr<uint8_t>(),
            output_gpu_batch_view.data_ptr<uint8_t>(),
            current_batch_rows, M, packed_cols
        );

        cudaDeviceSynchronize();
    }
}

// Función para descomprimir datos desde GPU y dejar el resultado en GPU
void unpack2bit_gpu_to_gpu(torch::Tensor input_gpu, torch::Tensor output_gpu) {
    TORCH_CHECK(input_gpu.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(output_gpu.device().is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(input_gpu.get_device() == output_gpu.get_device(), "Input and Output tensors must be on the same CUDA device");

    int N = output_gpu.size(0);
    int M = output_gpu.size(1);
    int packed_cols = (M + 3) / 4;

    TORCH_CHECK(input_gpu.size(0) == N, "Input tensor row dimension mismatch");
    TORCH_CHECK(input_gpu.size(1) == packed_cols, "Input tensor column dimension mismatch based on output shape");

    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks((packed_cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, N);

    unpack2bit_kernel<<<blocks, threads>>>(
        input_gpu.data_ptr<uint8_t>(),
        output_gpu.data_ptr<uint8_t>(),
        N, M, packed_cols
    );

    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack2bit_cpu_to_gpu", &pack2bit_cpu_to_gpu_cuda, "Pack 2-bit values from CPU tensor to GPU tensor (batched)");
    m.def("unpack2bit_gpu_to_gpu", &unpack2bit_gpu_to_gpu, "Unpack 2-bit values from GPU tensor to GPU tensor (batched)");
}