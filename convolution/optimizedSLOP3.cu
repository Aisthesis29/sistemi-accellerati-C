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

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

static inline int min_cpu(int a, int b) { return a < b ? a : b; }
static inline int max_cpu(int a, int b) { return a > b ? a : b; }

/* ============================================================
   CPU reference
   ============================================================ */
void bilateral_u8_rgb_cpu(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int radius, int sigma_s, int sigma_r)
{
    const int dim_kernel = 2 * radius + 1;
    const float inv_2_sigma_s2 = 1.f / (2.f * sigma_s * sigma_s);
    const float inv_2_sigma_r2 = 1.f / (2.f * sigma_r * sigma_r);

    float* space_weight = (float*)malloc(dim_kernel * dim_kernel * sizeof(float));
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
            space_weight[(dy+radius)*dim_kernel + (dx+radius)] =
                expf(-(dx*dx + dy*dy) * inv_2_sigma_s2);

    float color_weight[3*256];
    for (int i = 0; i < 3*256; ++i)
        color_weight[i] = expf(-(i*i) * inv_2_sigma_r2);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            int idx0 = (y * width + x) * 3;
            int cr = input[idx0+0];
            int cg = input[idx0+1];
            int cb = input[idx0+2];

            float wsum = 0.f, sr = 0.f, sg = 0.f, sb = 0.f;

            int y0 = max_cpu(0, y-radius);
            int y1 = min_cpu(height-1, y+radius);
            int x0 = max_cpu(0, x-radius);
            int x1 = min_cpu(width-1, x+radius);

            for (int yy = y0; yy <= y1; ++yy)
                for (int xx = x0; xx <= x1; ++xx) {

                    int idx = (yy * width + xx) * 3;
                    int vr = input[idx+0];
                    int vg = input[idx+1];
                    int vb = input[idx+2];

                    int dr = abs(vr-cr) + abs(vg-cg) + abs(vb-cb);
                    float w = space_weight[(yy-y+radius)*dim_kernel + (xx-x+radius)]
                              * color_weight[dr];

                    wsum += w;
                    sr += w * vr;
                    sg += w * vg;
                    sb += w * vb;
                }

            float invW = 1.f / wsum;
            output[idx0+0] = (unsigned char)(sr * invW + 0.5f);
            output[idx0+1] = (unsigned char)(sg * invW + 0.5f);
            output[idx0+2] = (unsigned char)(sb * invW + 0.5f);
        }
    }

    free(space_weight);
}

/* ============================================================
   GPU kernel – RGB only
   ============================================================ */
__global__ void bilateral_u8_rgb_ybase(
    const unsigned char* in,
    unsigned char* out,
    int width, int height,
    int y_base, int rows,
    int radius,
    const float* space_weight,
    const float* color_weight,
    int dim_kernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_local = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y_local >= rows) return;

    int y = y_base + y_local;
    if (y >= height) return;

    int idx0 = (y * width + x) * 3;
    int cr = in[idx0+0];
    int cg = in[idx0+1];
    int cb = in[idx0+2];

    float wsum = 0.f, sr = 0.f, sg = 0.f, sb = 0.f;

    int y0 = max(y-radius, 0);
    int y1 = min(y+radius, height-1);
    int x0 = max(x-radius, 0);
    int x1 = min(x+radius, width-1);

    for (int yy = y0; yy <= y1; ++yy)
        for (int xx = x0; xx <= x1; ++xx) {

            int idx = (yy * width + xx) * 3;
            int vr = in[idx+0];
            int vg = in[idx+1];
            int vb = in[idx+2];

            int dr = abs(vr-cr) + abs(vg-cg) + abs(vb-cb);
            float w = space_weight[(yy-y+radius)*dim_kernel + (xx-x+radius)]
                      * color_weight[dr];

            wsum += w;
            sr = fmaf(w, vr, sr);
            sg = fmaf(w, vg, sg);
            sb = fmaf(w, vb, sb);
        }

    float invW = 1.f / wsum;
    out[idx0+0] = (unsigned char)(sr * invW + 0.5f);
    out[idx0+1] = (unsigned char)(sg * invW + 0.5f);
    out[idx0+2] = (unsigned char)(sb * invW + 0.5f);
}

/* ============================================================
   MAIN
   ============================================================ */
int main(int argc, char** argv)
{
    if (argc < 5) {
        printf("usage: %s image kernel sigma_s sigma_r\n", argv[0]);
        return 1;
    }

    int dim_kernel = atoi(argv[2]);
    int sigma_s    = atoi(argv[3]);
    int sigma_r    = atoi(argv[4]);
    int radius     = dim_kernel / 2;

    int w, h, c;
    unsigned char* h_in = stbi_load(argv[1], &w, &h, &c, 3);
    if (!h_in) return 1;
    c = 3;

    size_t size = w * h * 3;
    unsigned char* h_out = (unsigned char*)malloc(size);
    unsigned char* h_cpu = (unsigned char*)malloc(size);

    unsigned char *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  size));
    CHECK(cudaMalloc(&d_out, size));
    CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    float* h_space = (float*)malloc(dim_kernel * dim_kernel * sizeof(float));
    for (int dy=-radius; dy<=radius; ++dy)
        for (int dx=-radius; dx<=radius; ++dx)
            h_space[(dy+radius)*dim_kernel + (dx+radius)] =
                expf(-(dx*dx + dy*dy)/(2.f*sigma_s*sigma_s));

    float h_color[3*256];
    for (int i=0;i<3*256;i++)
        h_color[i] = expf(-(i*i)/(2.f*sigma_r*sigma_r));

    float *d_space, *d_color;
    CHECK(cudaMalloc(&d_space, dim_kernel*dim_kernel*sizeof(float)));
    CHECK(cudaMalloc(&d_color, 3*256*sizeof(float)));
    CHECK(cudaMemcpy(d_space, h_space, dim_kernel*dim_kernel*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_color, h_color, 3*256*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid((w+15)/16, (h+15)/16);

    bilateral_u8_rgb_ybase<<<grid, block>>>(
        d_in, d_out,
        w, h,
        0, h,
        radius,
        d_space, d_color,
        dim_kernel
    );

    CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    stbi_write_png("risultato.png", w, h, 3, h_out, w*3);

    bilateral_u8_rgb_cpu(h_in, h_cpu, w, h, radius, sigma_s, sigma_r);

    printf("done\n");

    free(h_space);
    free(h_out);
    free(h_cpu);
    stbi_image_free(h_in);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_space);
    cudaFree(d_color);
    return 0;
}
