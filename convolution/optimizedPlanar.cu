// bilateral_planar.cu
// Build (example):
//   nvcc bilateral_planar.cu -O2 -o opt -gencode arch=compute_75,code=sm_75
//
// Run:
//   ./opt Images/img4.jpg 3 50 50
//
// Args:
//   <image_path> <kernel_size odd> <sigma_s int> <sigma_r int>

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// STB
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHECK(call)                                                         \
do {                                                                        \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                cudaGetErrorString(err));                                   \
        exit(1);                                                            \
    }                                                                       \
} while(0)

static inline int min_cpu(int a, int b) { return (a < b) ? a : b; }
static inline int max_cpu(int a, int b) { return (a > b) ? a : b; }

// -------------------- GPU kernels --------------------

__global__ void rgb_interleaved_to_planar(
    const unsigned char* __restrict__ inRGB,
    unsigned char* __restrict__ outR,
    unsigned char* __restrict__ outG,
    unsigned char* __restrict__ outB,
    int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int base = idx * channels;          // channels assumed >= 3

    outR[idx] = inRGB[base + 0];
    outG[idx] = inRGB[base + 1];
    outB[idx] = inRGB[base + 2];
}

__global__ void pack_planar_to_rgb(
    const unsigned char* __restrict__ inR,
    const unsigned char* __restrict__ inG,
    const unsigned char* __restrict__ inB,
    unsigned char* __restrict__ outRGB,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int o = idx * 3;
    outRGB[o + 0] = inR[idx];
    outRGB[o + 1] = inG[idx];
    outRGB[o + 2] = inB[idx];
}

// A single correct bilateral kernel over a y-range [y_base, y_base+rows).
// This matches the CPU reference (clamped window, no mirroring tricks).
__global__ void bilateral_planar_ybase(
    const unsigned char* __restrict__ inR,
    const unsigned char* __restrict__ inG,
    const unsigned char* __restrict__ inB,
    unsigned char* __restrict__ outR,
    unsigned char* __restrict__ outG,
    unsigned char* __restrict__ outB,
    int width, int height,
    int y_base, int rows,
    int radius,
    const float* __restrict__ space_weight,   // dim_kernel*dim_kernel, flattened
    const float* __restrict__ color_weight,   // (3*256) entries (dr in [0,765])
    int dim_kernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_local = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y_local >= rows) return;

    int y = y_base + y_local;
    if (y < 0 || y >= height) return;

    int idx0 = y * width + x;

    int center_r = (int)inR[idx0];
    int center_g = (int)inG[idx0];
    int center_b = (int)inB[idx0];

    int y0 = max(y - radius, 0);
    int yn = min(y + radius, height - 1);
    int x0 = max(x - radius, 0);
    int xn = min(x + radius, width - 1);

    float wsum  = 0.0f;
    float sum_r = 0.0f;
    float sum_g = 0.0f;
    float sum_b = 0.0f;

    for (int dy = y0; dy <= yn; ++dy) {
        int row = dy * width;
        int wy  = (dy - y + radius); // 0..dim_kernel-1
        for (int dx = x0; dx <= xn; ++dx) {
            int idx = row + dx;

            int vr = (int)inR[idx];
            int vg = (int)inG[idx];
            int vb = (int)inB[idx];

            int dr = abs(vr - center_r) + abs(vg - center_g) + abs(vb - center_b); // 0..765
            // Flattened space weight index: (kx * dim_kernel + ky)
            int kx = (dx - x + radius);
            float w = space_weight[kx * dim_kernel + wy] * color_weight[dr];

            wsum  += w;
            sum_r = fmaf(w, (float)vr, sum_r);
            sum_g = fmaf(w, (float)vg, sum_g);
            sum_b = fmaf(w, (float)vb, sum_b);
        }
    }

    float inv = 1.0f / wsum;
    outR[idx0] = (unsigned char)(sum_r * inv + 0.5f);
    outG[idx0] = (unsigned char)(sum_g * inv + 0.5f);
    outB[idx0] = (unsigned char)(sum_b * inv + 0.5f);
}

// -------------------- CPU reference --------------------

void bilateral_cpu_rgb_interleaved(
    const unsigned char* inRGB, unsigned char* outRGB,
    int width, int height, int radius, int sigma_s, int sigma_r)
{
    const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
    const float inv_2_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx0 = y * width + x;

            int center_r = (int)inRGB[idx0 * 3 + 0];
            int center_g = (int)inRGB[idx0 * 3 + 1];
            int center_b = (int)inRGB[idx0 * 3 + 2];

            int y0 = max_cpu(y - radius, 0);
            int yn = min_cpu(y + radius, height - 1);
            int x0 = max_cpu(x - radius, 0);
            int xn = min_cpu(x + radius, width - 1);

            float wsum  = 0.0f;
            float sum_r = 0.0f;
            float sum_g = 0.0f;
            float sum_b = 0.0f;

            for (int dy = y0; dy <= yn; ++dy) {
                for (int dx = x0; dx <= xn; ++dx) {
                    int idx = dy * width + dx;

                    int vr = (int)inRGB[idx * 3 + 0];
                    int vg = (int)inRGB[idx * 3 + 1];
                    int vb = (int)inRGB[idx * 3 + 2];

                    int ds2 = (dx - x) * (dx - x) + (dy - y) * (dy - y);
                    int dr  = abs(vr - center_r) + abs(vg - center_g) + abs(vb - center_b);
                    int dr2 = dr * dr;

                    float w_s = expf(-ds2 * inv_2_sigma_s2);
                    float w_r = expf(-dr2 * inv_2_sigma_r2);
                    float w   = w_s * w_r;

                    wsum  += w;
                    sum_r += w * vr;
                    sum_g += w * vg;
                    sum_b += w * vb;
                }
            }

            float inv = 1.0f / wsum;
            outRGB[idx0 * 3 + 0] = (unsigned char)(sum_r * inv + 0.5f);
            outRGB[idx0 * 3 + 1] = (unsigned char)(sum_g * inv + 0.5f);
            outRGB[idx0 * 3 + 2] = (unsigned char)(sum_b * inv + 0.5f);
        }
    }
}

bool verifyResults(const unsigned char* cpu, const unsigned char* gpu, int size, const char* label)
{
    int errors = 0;
    int grave  = 0;
    for (int i = 0; i < size; i++) {
        int diff = abs((int)cpu[i] - (int)gpu[i]);
        if (diff >= 1) {
            errors++;
            if (diff >= 2) grave++;
            if (errors < 200) {
                printf("Mismatch at %d: CPU=%d %s=%d (diff=%d)\n", i, cpu[i], label, gpu[i], diff);
            }
        }
    }
    if (errors) {
        printf("Total errors: %d / %d (%.3f%%)\n", errors, size, 100.0f * errors / size);
    }
    return grave == 0;
}

// -------------------- main --------------------

int main(int argc, char** argv)
{
    if (argc < 5) {
        printf("Usage: %s <image_path> <kernel_size odd> <sigma_s int> <sigma_r int>\n", argv[0]);
        return 1;
    }

    const char* inputFile = argv[1];
    int dim_kernel = atoi(argv[2]);
    int sigma_s    = atoi(argv[3]);
    int sigma_r    = atoi(argv[4]);

    if (dim_kernel < 1 || (dim_kernel % 2) == 0) {
        printf("kernel_size must be positive odd\n");
        return 2;
    }
    if (sigma_s <= 0 || sigma_r <= 0) {
        printf("sigma_s and sigma_r must be positive\n");
        return 2;
    }

    int radius = dim_kernel / 2;

    int width, height, channels_in;
    // Force 3 channels so everything is consistent.
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels_in, 3);
    if (!h_input) {
        printf("Error loading image %s\n", inputFile);
        return 1;
    }
    int channels = 3;
    printf("Image loaded: %dx%d (forced %d channels)\n", width, height, channels);

    size_t nPix = (size_t)width * (size_t)height;
    size_t imageSizeRGB = nPix * 3;

    unsigned char* h_outputGPU = (unsigned char*)malloc(imageSizeRGB);
    unsigned char* h_outputCPU = (unsigned char*)malloc(imageSizeRGB);
    if (!h_outputGPU || !h_outputCPU) {
        printf("Host malloc failed\n");
        return 1;
    }

    // ---------- weights on host ----------
    float* h_space = (float*)malloc((size_t)dim_kernel * (size_t)dim_kernel * sizeof(float));
    float* h_color = (float*)malloc((size_t)(3 * 256) * sizeof(float));
    if (!h_space || !h_color) {
        printf("Host malloc weights failed\n");
        return 1;
    }

    const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
    for (int kx = -radius; kx <= radius; ++kx) {
        for (int ky = -radius; ky <= radius; ++ky) {
            int ds2 = kx * kx + ky * ky;
            float w = expf(-ds2 * inv_2_sigma_s2);
            int ix = kx + radius;
            int iy = ky + radius;
            h_space[ix * dim_kernel + iy] = w; // same flattening as kernel
        }
    }

    const float inv_2_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);
    for (int i = 0; i < 3 * 256; ++i) {
        h_color[i] = expf(-(float)(i * i) * inv_2_sigma_r2);
    }

    // ---------- device allocations ----------
    unsigned char *d_inputRGB = nullptr;
    unsigned char *d_inR = nullptr, *d_inG = nullptr, *d_inB = nullptr;
    unsigned char *d_outR = nullptr, *d_outG = nullptr, *d_outB = nullptr;
    unsigned char *d_outputRGB = nullptr;

    float *d_space = nullptr, *d_color = nullptr;

    CHECK(cudaMalloc((void**)&d_inputRGB, imageSizeRGB));
    CHECK(cudaMalloc((void**)&d_inR, nPix));
    CHECK(cudaMalloc((void**)&d_inG, nPix));
    CHECK(cudaMalloc((void**)&d_inB, nPix));

    CHECK(cudaMalloc((void**)&d_outR, nPix));
    CHECK(cudaMalloc((void**)&d_outG, nPix));
    CHECK(cudaMalloc((void**)&d_outB, nPix));

    CHECK(cudaMalloc((void**)&d_outputRGB, imageSizeRGB));

    CHECK(cudaMalloc((void**)&d_space, (size_t)dim_kernel * (size_t)dim_kernel * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_color, (size_t)(3 * 256) * sizeof(float)));

    CHECK(cudaMemcpy(d_inputRGB, h_input, imageSizeRGB, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_space, h_space, (size_t)dim_kernel * (size_t)dim_kernel * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_color, h_color, (size_t)(3 * 256) * sizeof(float), cudaMemcpyHostToDevice));

    // ---------- launch config ----------
    int blockSize = 16;
    dim3 block(blockSize, blockSize);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Split input to planar
    rgb_interleaved_to_planar<<<grid, block>>>(d_inputRGB, d_inR, d_inG, d_inB, width, height, channels);
    CHECK(cudaGetLastError());

    // ---------- bilateral (3-region launch like your structure) ----------
    int rows_border = dim_kernel;
    int inner_rows = height - 2 * rows_border;
    if (inner_rows < 0) inner_rows = 0;

    // top border
    {
        dim3 grid_row((width + block.x - 1) / block.x,
                      (rows_border + block.y - 1) / block.y);
        bilateral_planar_ybase<<<grid_row, block>>>(
            d_inR, d_inG, d_inB,
            d_outR, d_outG, d_outB,
            width, height,
            0, rows_border,
            radius, d_space, d_color, dim_kernel
        );
        CHECK(cudaGetLastError());
    }

    // inner
    if (inner_rows > 0) {
        dim3 grid_inner((width + block.x - 1) / block.x,
                        (inner_rows + block.y - 1) / block.y);
        bilateral_planar_ybase<<<grid_inner, block>>>(
            d_inR, d_inG, d_inB,
            d_outR, d_outG, d_outB,
            width, height,
            rows_border, inner_rows,
            radius, d_space, d_color, dim_kernel
        );
        CHECK(cudaGetLastError());
    }

    // bottom border
    {
        int y_base = rows_border + inner_rows;
        dim3 grid_row((width + block.x - 1) / block.x,
                      (rows_border + block.y - 1) / block.y);
        bilateral_planar_ybase<<<grid_row, block>>>(
            d_inR, d_inG, d_inB,
            d_outR, d_outG, d_outB,
            width, height,
            y_base, rows_border,
            radius, d_space, d_color, dim_kernel
        );
        CHECK(cudaGetLastError());
    }

    // Pack planar output back to interleaved RGB for PNG + verify
    pack_planar_to_rgb<<<grid, block>>>(d_outR, d_outG, d_outB, d_outputRGB, width, height);
    CHECK(cudaGetLastError());

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_outputGPU, d_outputRGB, imageSizeRGB, cudaMemcpyDeviceToHost));

    // Save GPU result
    stbi_write_png("risultato.png", width, height, 3, h_outputGPU, width * 3);
    printf("Saved: risultato.png\n");

    // CPU reference + verify
    bilateral_cpu_rgb_interleaved(h_input, h_outputCPU, width, height, radius, sigma_s, sigma_r);
    bool ok = verifyResults(h_outputCPU, h_outputGPU, (int)imageSizeRGB, "GPU");
    printf("%s\n", ok ? "✓ PASS" : "✗ FAIL");

    // cleanup
    stbi_image_free(h_input);
    free(h_outputGPU);
    free(h_outputCPU);
    free(h_space);
    free(h_color);

    CHECK(cudaFree(d_inputRGB));
    CHECK(cudaFree(d_inR));
    CHECK(cudaFree(d_inG));
    CHECK(cudaFree(d_inB));
    CHECK(cudaFree(d_outR));
    CHECK(cudaFree(d_outG));
    CHECK(cudaFree(d_outB));
    CHECK(cudaFree(d_outputRGB));
    CHECK(cudaFree(d_space));
    CHECK(cudaFree(d_color));

    return ok ? 0 : 3;
}
