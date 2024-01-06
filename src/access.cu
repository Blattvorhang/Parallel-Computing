__global__ void computeLogSqrt(float* data, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        data[i] = logf(sqrtf(data[i]));
    }
}

void applyLogSqrt(const float* data, int len) {
    float* dev_data;
    cudaMalloc((void**)&dev_data, len * sizeof(float));
    cudaMemcpy(dev_data, data, len * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;  // 可以根据需要调整
    int numBlocks = (len + blockSize - 1) / blockSize;
    computeLogSqrt<<<numBlocks, blockSize>>>(dev_data, len);

    cudaFree(dev_data);
}