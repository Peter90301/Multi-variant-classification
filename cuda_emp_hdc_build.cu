#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

__global__ void emp_hdc_accumulate_kernel(
    int nnz,
    int active_dims,
    int dim,
    const int *nnz_sample_idx,
    const int *nnz_obs_idx,
    const float *nnz_values,
    const int *obs_cols,
    const float *obs_signs,
    float *matrix
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nnz * active_dims;
    if (tid >= total) return;

    int nz = tid / active_dims;
    int a = tid - nz * active_dims;
    int sample = nnz_sample_idx[nz];
    int obs = nnz_obs_idx[nz];
    int col = obs_cols[obs * active_dims + a];
    float value = nnz_values[nz] * obs_signs[obs * active_dims + a];
    atomicAdd(matrix + sample * dim + col, value);
}

int build_emp_hdc_matrix(
    int device,
    int n_samples,
    int dim,
    int nnz,
    int n_obs,
    int active_dims,
    const int *h_nnz_sample_idx,
    const int *h_nnz_obs_idx,
    const float *h_nnz_values,
    const int *h_obs_cols,
    const float *h_obs_signs,
    float *h_matrix,
    float *elapsed_ms
) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return (int)err;

    int *d_nnz_sample_idx = nullptr;
    int *d_nnz_obs_idx = nullptr;
    float *d_nnz_values = nullptr;
    int *d_obs_cols = nullptr;
    float *d_obs_signs = nullptr;
    float *d_matrix = nullptr;
    cudaEvent_t start, stop;

    size_t nnz_int_bytes = (size_t)nnz * sizeof(int);
    size_t nnz_float_bytes = (size_t)nnz * sizeof(float);
    size_t obs_int_bytes = (size_t)n_obs * active_dims * sizeof(int);
    size_t obs_float_bytes = (size_t)n_obs * active_dims * sizeof(float);
    size_t matrix_bytes = (size_t)n_samples * dim * sizeof(float);

    cudaMalloc((void **)&d_nnz_sample_idx, nnz_int_bytes);
    cudaMalloc((void **)&d_nnz_obs_idx, nnz_int_bytes);
    cudaMalloc((void **)&d_nnz_values, nnz_float_bytes);
    cudaMalloc((void **)&d_obs_cols, obs_int_bytes);
    cudaMalloc((void **)&d_obs_signs, obs_float_bytes);
    cudaMalloc((void **)&d_matrix, matrix_bytes);
    err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;

    cudaMemset(d_matrix, 0, matrix_bytes);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(d_nnz_sample_idx, h_nnz_sample_idx, nnz_int_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnz_obs_idx, h_nnz_obs_idx, nnz_int_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnz_values, h_nnz_values, nnz_float_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_cols, h_obs_cols, obs_int_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_signs, h_obs_signs, obs_float_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int total = nnz * active_dims;
    int blocks = (total + threads - 1) / threads;
    emp_hdc_accumulate_kernel<<<blocks, threads>>>(
        nnz, active_dims, dim,
        d_nnz_sample_idx, d_nnz_obs_idx, d_nnz_values,
        d_obs_cols, d_obs_signs, d_matrix
    );

    cudaMemcpy(h_matrix, d_matrix, matrix_bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_nnz_sample_idx);
    cudaFree(d_nnz_obs_idx);
    cudaFree(d_nnz_values);
    cudaFree(d_obs_cols);
    cudaFree(d_obs_signs);
    cudaFree(d_matrix);
    return (int)cudaGetLastError();
}

}
