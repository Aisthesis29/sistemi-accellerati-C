// bilateral_rgba_single_file.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

// ===== STB (single-file) =====
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ===== CONFIG =====
#define DIM 45                      // max supported kernel size (odd). dim_kernel MUST be <= DIM
__constant__ float space_weight[DIM * DIM];

static inline int min_cpu(int v1, int v2) { return (v1 > v2) ? v2 : v1; }
static inline int max_cpu(int v1, int v2) { return (v1 < v2) ? v2 : v1; }

#define CHECK(call)                                                          \
{                                                                            \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess)                                                \
    {                                                                        \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);               \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                             \
    }                                                                        \
}

// ===== GPU KERNELS =====

// Convert input RGB (3 bytes/pixel) to RGBA (uchar4)
__global__ void rgb_to_rgba(const unsigned char *d_input_rgb, uchar4 *rgba, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixel = y * width + x;
    int idx = pixel * 3;

    unsigned char r = d_input_rgb[idx + 0];
    unsigned char g = d_input_rgb[idx + 1];
    unsigned char b = d_input_rgb[idx + 2];

    rgba[pixel] = make_uchar4(r, g, b, 255);
}

// Bilateral filter: input RGBA, output RGBA (1 store per pixel)
__global__ void bilateral_rgba(
    const uchar4 *rgba,
    uchar4 *out_rgba,
    int width,
    int height,
    int borders,
    int rows,
    int radius,
    const float *color_weight,
    int dim_kernel
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_local = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y_local >= height) return;

    int x0 = max(x - radius, 0);
    int xn = min(x + radius, width - 1);

    float wsum = 0.0f;
    float sum_r = 0.0f;
    float sum_g = 0.0f;
    float sum_b = 0.0f;

    int idx0; // pixel index (y*width + x)

    if (y_local >= borders) {
        int y = y_local - radius;
        idx0 = y * width + x;

        uchar4 center = rgba[idx0];
        unsigned char center_r = center.x;
        unsigned char center_g = center.y;
        unsigned char center_b = center.z;

        int i = 0;
        float w, w_s;

        for (int dy = y - radius; dy < y; ++dy) {
            for (int dx = x0; dx <= xn; ++dx) {
                int idxUpp  = dy * width + dx;
                int idxDown = (dy + dim_kernel - 1 - 2 * (i)) * width + dx;

                uchar4 valUpp  = rgba[idxUpp];
                uchar4 valDown = rgba[idxDown];

                w_s = space_weight[(dx - x + radius) * dim_kernel + (dy - y + radius)];

                int drU = __sad(valUpp.x, center_r,
                         __sad(valUpp.y, center_g,
                          __sad(valUpp.z, center_b, 0)));
                w = w_s * color_weight[drU];
                wsum += w;
                sum_r = fmaf(w, (float)valUpp.x, sum_r);
                sum_g = fmaf(w, (float)valUpp.y, sum_g);
                sum_b = fmaf(w, (float)valUpp.z, sum_b);

                int drD = __sad(valDown.x, center_r,
                         __sad(valDown.y, center_g,
                          __sad(valDown.z, center_b, 0)));
                w = w_s * color_weight[drD];
                wsum += w;
                sum_r = fmaf(w, (float)valDown.x, sum_r);
                sum_g = fmaf(w, (float)valDown.y, sum_g);
                sum_b = fmaf(w, (float)valDown.z, sum_b);
            }
            i++;
        }

        for (int dx = x0; dx <= xn; dx++) {
            int idx = y * width + dx;
            uchar4 val = rgba[idx];

            int dr = __sad(val.x, center_r,
                     __sad(val.y, center_g,
                      __sad(val.z, center_b, 0)));
            float w2 = space_weight[(dx - x + radius) * dim_kernel + radius] * color_weight[dr];

            wsum += w2;
            sum_r = fmaf(w2, (float)val.x, sum_r);
            sum_g = fmaf(w2, (float)val.y, sum_g);
            sum_b = fmaf(w2, (float)val.z, sum_b);
        }
    }
    else if (y_local < radius) {
        idx0 = y_local * width + x;

        uchar4 center = rgba[idx0];
        unsigned char center_r = center.x;
        unsigned char center_g = center.y;
        unsigned char center_b = center.z;

        for (int dy = 0; dy <= y_local + radius; ++dy) {
            for (int dx = x0; dx <= xn; ++dx) {
                int idx = dy * width + dx;
                uchar4 val = rgba[idx];

                int dr = __sad(val.x, center_r,
                         __sad(val.y, center_g,
                          __sad(val.z, center_b, 0)));
                float w = space_weight[(dx - x + radius) * dim_kernel + (dy - y_local + radius)] * color_weight[dr];

                wsum += w;
                sum_r = fmaf(w, (float)val.x, sum_r);
                sum_g = fmaf(w, (float)val.y, sum_g);
                sum_b = fmaf(w, (float)val.z, sum_b);
            }
        }
    }
    else {
        int y = rows + y_local;
        idx0 = y * width + x;

        uchar4 center = rgba[idx0];
        unsigned char center_r = center.x;
        unsigned char center_g = center.y;
        unsigned char center_b = center.z;

        for (int dy = y - radius; dy <= height - 1; ++dy) {
            for (int dx = x0; dx <= xn; ++dx) {
                int idx = dy * width + dx;
                uchar4 val = rgba[idx];

                int dr = __sad(val.x, center_r,
                         __sad(val.y, center_g,
                          __sad(val.z, center_b, 0)));
                float w = space_weight[(dx - x + radius) * dim_kernel + (dy - y + radius)] * color_weight[dr];

                wsum += w;
                sum_r = fmaf(w, (float)val.x, sum_r);
                sum_g = fmaf(w, (float)val.y, sum_g);
                sum_b = fmaf(w, (float)val.z, sum_b);
            }
        }
    }

    float inv_Wsum = 1.0f / wsum;
    unsigned char r = (unsigned char)(sum_r * inv_Wsum + 0.5f);
    unsigned char g = (unsigned char)(sum_g * inv_Wsum + 0.5f);
    unsigned char b = (unsigned char)(sum_b * inv_Wsum + 0.5f);

    out_rgba[idx0] = make_uchar4(r, g, b, 255);
}

// ===== CPU REFERENCE (RGB) =====
void bilateral_u8_rgb_cpu(unsigned char *h_input, unsigned char *out, int width, int height, int radius, int sigma_s, int sigma_r) {
    if (!out || !h_input || width <= 0 || height <= 0 || radius < 0) return;
    if (sigma_s <= 0 || sigma_r <= 0) return;

    int y0, yn, x0, xn;
    const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
    const float inv_2_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx0 = y * width + x;

            int center_r = (int)h_input[idx0 * 3 + 0];
            int center_g = (int)h_input[idx0 * 3 + 1];
            int center_b = (int)h_input[idx0 * 3 + 2];

            float wsum = 0.0f;
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;

            y0 = max_cpu(y - radius, 0);
            yn = min_cpu(y + radius, height - 1);
            x0 = max_cpu(x - radius, 0);
            xn = min_cpu(x + radius, width - 1);

            for (int dy = y0; dy <= yn; ++dy) {
                for (int dx = x0; dx <= xn; ++dx) {
                    int idx = dy * width + dx;

                    int val_r = (int)h_input[idx * 3 + 0];
                    int val_g = (int)h_input[idx * 3 + 1];
                    int val_b = (int)h_input[idx * 3 + 2];

                    int ds2 = (dx - x) * (dx - x) + (dy - y) * (dy - y);
                    int dr  = abs(val_r - center_r) + abs(val_g - center_g) + abs(val_b - center_b);
                    int dr2 = dr * dr;

                    float w_s = expf(-ds2 * inv_2_sigma_s2);
                    float w_r = expf(-dr2 * inv_2_sigma_r2);
                    float w   = w_s * w_r;

                    wsum += w;
                    sum_r += w * val_r;
                    sum_g += w * val_g;
                    sum_b += w * val_b;
                }
            }

            float inv_Wsum = 1.0f / wsum;
            out[idx0 * 3 + 0] = (unsigned char)(sum_r * inv_Wsum + 0.5f);
            out[idx0 * 3 + 1] = (unsigned char)(sum_g * inv_Wsum + 0.5f);
            out[idx0 * 3 + 2] = (unsigned char)(sum_b * inv_Wsum + 0.5f);
        }
    }
}

// ===== VERIFY (CPU RGB vs GPU RGBA) =====
bool verifyRGBvsRGBA(const unsigned char* cpu_rgb, const unsigned char* gpu_rgba, int width, int height) {
    int pixels = width * height;
    int errors = 0;
    int grave = 0;

    for (int i = 0; i < pixels; i++) {
        int cpu_idx = i * 3;
        int gpu_idx = i * 4;

        for (int c = 0; c < 3; c++) {
            int diff = abs((int)cpu_rgb[cpu_idx + c] - (int)gpu_rgba[gpu_idx + c]);
            if (diff >= 1) {
                errors++;
                if (diff >= 2) grave++;
                if (grave < 20 && diff > 1) {
                    printf("Mismatch pixel %d ch %d: CPU=%d GPU=%d (diff=%d)\n",
                           i, c, cpu_rgb[cpu_idx + c], gpu_rgba[gpu_idx + c], diff);
                }
            }
        }

        if (gpu_rgba[gpu_idx + 3] != 255) {
            grave++;
            if (grave < 20) printf("Alpha mismatch pixel %d: %d\n", i, gpu_rgba[gpu_idx + 3]);
        }
    }

    if (errors > 0) {
        printf("Total errors (RGB only): %d / %d (%.3f%%)\n",
               errors, pixels * 3, 100.0f * (float)errors / (float)(pixels * 3));
    }

    return grave == 0;
}

// ===== MAIN =====
int main(int argc, char **argv) {
    if (argc < 7) {
        printf("Usage: %s <input_image> <kernel_size> <sigma_s> <sigma_r> <block_x> <block_y>\n", argv[0]);
        printf("kernel_size: dimensione matrice (odd, <= %d)\n", DIM);
        printf("sigma_s, sigma_r: interi positivi\n");
        return 1;
    }

    const char* inputFile = argv[1];
    int dim_kernel = atoi(argv[2]);
    int sigma_s = atoi(argv[3]);
    int sigma_r = atoi(argv[4]);
    int block_x = atoi(argv[5]);
    int block_y = atoi(argv[6]);

    if (dim_kernel % 2 == 0 || dim_kernel < 1 || dim_kernel > DIM) {
        printf("kernel_size non valido: deve essere dispari, 1..%d\n", DIM);
        return 2;
    }
    if (sigma_s <= 0 || sigma_r <= 0) {
        printf("sigma_s e sigma_r devono essere > 0\n");
        return 2;
    }
    if (block_x <= 0 || block_y <= 0) {
        printf("block_x e block_y devono essere > 0\n");
        return 2;
    }

    int radius = dim_kernel / 2;

    // ===== Load image =====
    int width, height, channels;
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels, 0);
    if (!h_input) {
        printf("Error loading image %s\n", inputFile);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

    if (channels != 3) {
        printf("Questo file assume input RGB (3 canali). Canali trovati=%d\n", channels);
        stbi_image_free(h_input);
        return 1;
    }

    const int imageSizeRGB  = width * height * 3;
    const int imageSizeRGBA = width * height * 4;

    // ===== Host output (RGBA bytes) =====
    unsigned char* h_output_rgba = (unsigned char*)malloc(imageSizeRGBA);
    if (!h_output_rgba) {
        printf("malloc failed (h_output_rgba)\n");
        stbi_image_free(h_input);
        return 1;
    }

    // ===== Device buffers =====
    unsigned char *d_input_rgb = NULL;
    uchar4 *d_rgba_in = NULL;
    uchar4 *d_rgba_out = NULL;

    CHECK(cudaMalloc((void**)&d_input_rgb, imageSizeRGB));
    CHECK(cudaMalloc((void**)&d_rgba_in,  width * height * sizeof(uchar4)));
    CHECK(cudaMalloc((void**)&d_rgba_out, width * height * sizeof(uchar4)));

    CHECK(cudaMemcpy(d_input_rgb, h_input, imageSizeRGB, cudaMemcpyHostToDevice));

    dim3 block(block_x, block_y);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // RGB -> RGBA
    rgb_to_rgba<<<grid, block>>>(d_input_rgb, d_rgba_in, width, height);
    CHECK(cudaGetLastError());

    // ===== Prepare weights =====
    const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);

    float *h_space_weight = (float*)malloc(dim_kernel * dim_kernel * sizeof(float));
    if (!h_space_weight) {
        printf("malloc failed (h_space_weight)\n");
        stbi_image_free(h_input);
        free(h_output_rgba);
        CHECK(cudaFree(d_input_rgb));
        CHECK(cudaFree(d_rgba_in));
        CHECK(cudaFree(d_rgba_out));
        return 1;
    }

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int ds2 = (i * i) + (j * j);
            h_space_weight[(i + radius) * dim_kernel + (j + radius)] = expf(-ds2 * inv_2_sigma_s2);
        }
    }
    CHECK(cudaMemcpyToSymbol(space_weight, h_space_weight, dim_kernel * dim_kernel * sizeof(float)));
    free(h_space_weight);

    const float inv_2_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);
    float color_weight[3 * 256];
    for (int i = 0; i < 256 * 3; i++) {
        color_weight[i] = expf((float)(i * i) * -inv_2_sigma_r2);
    }

    float *d_color_weight = NULL;
    CHECK(cudaMalloc((void**)&d_color_weight, sizeof(float) * 3 * 256));
    CHECK(cudaMemcpy(d_color_weight, color_weight, sizeof(float) * 3 * 256, cudaMemcpyHostToDevice));

    int inner_rows = height - (2 * radius);

    // ===== Bilateral (RGBA output) =====
    bilateral_rgba<<<grid, block>>>(
        d_rgba_in,
        d_rgba_out,
        width, height,
        2 * radius,
        inner_rows,
        radius,
        d_color_weight,
        dim_kernel
    );
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // ===== Copy back RGBA bytes =====
    CHECK(cudaMemcpy(h_output_rgba, d_rgba_out, imageSizeRGBA, cudaMemcpyDeviceToHost));

    // ===== Save PNG RGBA =====
    stbi_write_png("risultato.png", width, height, 4, h_output_rgba, width * 4);
    printf("Saved risultato.png (RGBA)\n");

    // ===== CPU reference + verify (optional) =====
    unsigned char* h_output_cpu_rgb = (unsigned char*)malloc(imageSizeRGB);
    if (!h_output_cpu_rgb) {
        printf("malloc failed (h_output_cpu_rgb)\n");
    } else {
        bilateral_u8_rgb_cpu(h_input, h_output_cpu_rgb, width, height, radius, sigma_s, sigma_r);
        bool correct = verifyRGBvsRGBA(h_output_cpu_rgb, h_output_rgba, width, height);
        if (correct) printf("✓ Test PASSATO: CPU(RGB) e GPU(RGBA) matchano su RGB (alpha=255)\n");
        else printf("✗ Test FALLITO\n");
        free(h_output_cpu_rgb);
    }

    // ===== Cleanup =====
    stbi_image_free(h_input);
    free(h_output_rgba);
    CHECK(cudaFree(d_input_rgb));
    CHECK(cudaFree(d_rgba_in));
    CHECK(cudaFree(d_rgba_out));
    CHECK(cudaFree(d_color_weight));

    return 0;
}
