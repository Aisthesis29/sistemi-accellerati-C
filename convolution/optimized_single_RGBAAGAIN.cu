#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

// Include STB image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define DEBUG_IDX 12384
#define DIM 45

__constant__ float space_weight[DIM * DIM];

static inline int min_cpu(int v1, int v2) { return (v1 > v2) ? v2 : v1; }
static inline int max_cpu(int v1, int v2) { return (v1 < v2) ? v2 : v1; }

#define CHECK(call)                                                        \
  {                                                                        \
    const cudaError_t error = call;                                        \
    if (error != cudaSuccess) {                                            \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);               \
      fprintf(stderr, "code: %d, reason: %s\n", (int)error, cudaGetErrorString(error)); \
      exit(1);                                                             \
    }                                                                      \
  }

// Funzione per verificare correttezza
bool verifyResults(unsigned char* cpu_result, unsigned char* gpu_result, int size, const char* label)
{
    int errors = 0;
    int grave_errors = 0;
    for (int i = 0; i < size; i++) {
        int diff = abs((int)cpu_result[i] - (int)gpu_result[i]);
        if (diff >= 1) {
            errors++;
            if (diff >= 2) grave_errors++;
            if (grave_errors < 20 && diff > 1) {
                printf("Mismatch at index %d: CPU=%d, %s=%d (diff=%d)\n",
                       i, cpu_result[i], label, gpu_result[i], diff);
            }
        }
    }

    if (errors > 0) {
        printf("Total errors: %d / %d (%.3f%%)\n",
               errors, size, 100.0f * errors / size);
    }

    return grave_errors == 0;
}

__global__ void bilateral_u8_gray(uchar4 *rgba, unsigned char *out, int width, int height,int borders,int rows, int radius, float *color_weight, int dim_kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_local = blockIdx.y * blockDim.y + threadIdx.y;
        
    if (x >= width || y_local >= height) return;
    int x0, xn, idx0;
    x0 = max(x-radius, 0);
    xn = min(x+radius, width-1);
    
    float wsum = 0.0f;
    float sum_r = 0.0f;
    float sum_g = 0.0f;
    float sum_b = 0.0f;

    if(y_local >= borders) {
        int y = y_local - radius;
        idx0 = y * width + x;

        uchar4 pixel = rgba[idx0];
        int center_r = (int)pixel.x;
        int center_g = (int)pixel.y;
        int center_b = (int)pixel.z;
        
        int i=0;
        float w,w_s;
        for (int dy = y-radius; dy < y; ++dy) {
            for (int dx = x0; dx <= xn; ++dx) {
                int idxUpp = dy*width+dx;
                int idxDown = (dy+dim_kernel-1-2*(i))*width+dx;

                uchar4 valUpp = rgba[idxUpp];
                int val_rU = (int)valUpp.x;
                int val_gU = (int)valUpp.y;
                int val_bU = (int)valUpp.z;
                uchar4 valDown = rgba[idxDown];
                int val_rD = (int)valDown.x;
                int val_gD = (int)valDown.y;
                int val_bD = (int)valDown.z;

                w_s = space_weight[(dx-x+radius)*dim_kernel+(dy-y+radius)];
                int dr = __sad(val_rU, center_r,
         __sad(val_gU, center_g,
               abs(val_bU - center_b)));
                w = w_s * color_weight[dr];
                wsum += w;
                sum_r = fmaf(w, val_rU, sum_r);  //sum_r += w * val_r;
                sum_g = fmaf(w, val_gU, sum_g);  //sum_g += w * val_g;
                sum_b = fmaf(w, val_bU, sum_b);  //sum_b += w * val_b;

                int drD = __sad(val_rD, center_r,
         __sad(val_gD, center_g,
               abs(val_bD - center_b)));
                w = w_s * color_weight[drD];
                wsum += w;
                sum_r = fmaf(w, val_rD, sum_r);  //sum_r += w * val_r;
                sum_g = fmaf(w, val_gD, sum_g);  //sum_g += w * val_g;
                sum_b = fmaf(w, val_bD, sum_b);  //sum_b += w * val_b;
            }
            i++;
        }
            
        for(int dx=x0; dx<=xn; dx++) {
            int idx = y * width + dx;
            uchar4 val = rgba[idx];
            int val_r = (int)val.x;
            int val_g = (int)val.y;
            int val_b = (int)val.z;

            int dr = __sad(val_r, center_r,
         __sad(val_g, center_g,
               abs(val_b - center_b)));
            float w = space_weight[(dx-x+radius)*dim_kernel+radius] * color_weight[dr];

            wsum += w;
            sum_r = fmaf(w, val_r, sum_r);  //sum_r += w * val_r;
            sum_g = fmaf(w, val_g, sum_g);  //sum_g += w * val_g;
            sum_b = fmaf(w, val_b, sum_b);  //sum_b += w * val_b;      
        }
    } else if(y_local < radius) {
        idx0 = y_local * width + x;
        uchar4 pixel = rgba[idx0];
        int center_r = (int)pixel.x;
        int center_g = (int)pixel.y;
        int center_b = (int)pixel.z;

        for (int dy = 0; dy <= y_local+radius; ++dy) {
            for (int dx = x0; dx <= xn; ++dx) {
                int idx = dy * width + dx;

                uchar4 val = rgba[idx];
                int val_r = (int)val.x;
                int val_g = (int)val.y;
                int val_b = (int)val.z;

                int dr = __sad(val_r, center_r,
         __sad(val_g, center_g,
               abs(val_b - center_b)));
                float w = space_weight[(dx-x+radius)*dim_kernel+(dy-y_local+radius)] * color_weight[dr];

                wsum += w;
                sum_r = fmaf(w, val_r, sum_r);  //sum_r += w * val_r;
                sum_g = fmaf(w, val_g, sum_g);  //sum_g += w * val_g;
                sum_b = fmaf(w, val_b, sum_b);  //sum_b += w * val_b;
            }
        }
    } else {
        int y = rows + y_local;            
        idx0 = y * width +x;
        uchar4 pixel = rgba[idx0];
        int center_r = (int)pixel.x;
        int center_g = (int)pixel.y;
        int center_b = (int)pixel.z;

        for (int dy = y-radius; dy <= height-1; ++dy) {
            for (int dx = x0; dx <= xn; ++dx) {
                int idx = dy * width + dx;

                uchar4 val = rgba[idx];
                int val_r = (int)val.x;
                int val_g = (int)val.y;
                int val_b = (int)val.z;

               int dr = __sad(val_r, center_r,
         __sad(val_g, center_g,
               abs(val_b - center_b)));
                float w = space_weight[(dx-x+radius)*dim_kernel+(dy-y+radius)] * color_weight[dr];

                wsum += w;
                sum_r = fmaf(w, val_r, sum_r);  //sum_r += w * val_r;
                sum_g = fmaf(w, val_g, sum_g);  //sum_g += w * val_g;
                sum_b = fmaf(w, val_b, sum_b);  //sum_b += w * val_b;
            }
        }
    }
    
    float inv_Wsum = 1.f/wsum;
    out[idx0*3] = (unsigned char)(sum_r*inv_Wsum+0.5f);
    out[idx0*3+1] = (unsigned char)(sum_g*inv_Wsum+0.5f);
    out[idx0*3+2] = (unsigned char)(sum_b*inv_Wsum+0.5f);
}

void bilateral_u8_gray_cpu(const unsigned char* rgba_in, unsigned char* out_rgb,
                           int width, int height, int radius, int sigma_s, int sigma_r)
{
    if (!out_rgb || !rgba_in || width <= 0 || height <= 0 || radius < 0) return;
    if (sigma_s <= 0 || sigma_r <= 0) return;

    const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
    const float inv_2_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx0 = y * width + x;
            int c4 = idx0 * 4;

            int center_r = (int)rgba_in[c4 + 0];
            int center_g = (int)rgba_in[c4 + 1];
            int center_b = (int)rgba_in[c4 + 2];

            float wsum = 0.0f;
            float sum_r = 0.0f;
            float sum_g = 0.0f;
            float sum_b = 0.0f;

            int y0 = max_cpu(y - radius, 0);
            int yn = min_cpu(y + radius, height - 1);
            int x0 = max_cpu(x - radius, 0);
            int xn = min_cpu(x + radius, width - 1);

            for (int dy = y0; dy <= yn; ++dy) {
                for (int dx = x0; dx <= xn; ++dx) {
                    int idx = dy * width + dx;
                    int b4 = idx * 4;

                    int val_r = (int)rgba_in[b4 + 0];
                    int val_g = (int)rgba_in[b4 + 1];
                    int val_b = (int)rgba_in[b4 + 2];

                    int ds2 = (dx - x) * (dx - x) + (dy - y) * (dy - y);
                    int dr  = abs(val_r - center_r) + abs(val_g - center_g) + abs(val_b - center_b);
                    int dr2 = dr * dr;

                    float w_s = expf(-ds2 * inv_2_sigma_s2);
                    float w_r = expf(-dr2 * inv_2_sigma_r2);
                    float w   = w_s * w_r;

                    wsum += w;
                    sum_r += w * (float)val_r;
                    sum_g += w * (float)val_g;
                    sum_b += w * (float)val_b;
                }
            }

            float inv_Wsum = 1.f / wsum;
            int o3 = idx0 * 3;
            out_rgb[o3 + 0] = (unsigned char)(sum_r * inv_Wsum + 0.5f);
            out_rgb[o3 + 1] = (unsigned char)(sum_g * inv_Wsum + 0.5f);
            out_rgb[o3 + 2] = (unsigned char)(sum_b * inv_Wsum + 0.5f);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 7) {
        printf("Usage: %s <input.png/jpg> <kernel_size> <sigma_s> <sigma_r> <block_x> <block_y>\n", argv[0]);
        return 1;
    }

    const char* inputFile = argv[1];
    int dim_kernel = atoi(argv[2]);
    int sigma_s = atoi(argv[3]);
    int sigma_r = atoi(argv[4]);
    int block_x = atoi(argv[5]);
    int block_y = atoi(argv[6]);

    if (dim_kernel % 2 == 0 || dim_kernel < 1) {
        printf("kernel_size non valido (deve essere dispari e >0)\n");
        return 2;
    }
    if (dim_kernel > DIM) {
        printf("kernel_size=%d supera DIM=%d (aumenta DIM o riduci kernel)\n", dim_kernel, DIM);
        return 2;
    }
    if (sigma_s <= 0 || sigma_r <= 0) {
        printf("sigma_s e sigma_r devono essere >0\n");
        return 2;
    }
    if (block_x <= 0 || block_y <= 0) {
        printf("block_x e block_y devono essere >0\n");
        return 2;
    }

    int radius = dim_kernel / 2;

    // ========== Caricamento immagine (FORZA RGBA) ==========
    int width, height, channels_in_file;
    const int IN_COMP  = 4; // RGBA
    const int OUT_COMP = 3; // RGB

    unsigned char* h_input_rgba = stbi_load(inputFile, &width, &height, &channels_in_file, IN_COMP);
    if (!h_input_rgba) {
        printf("Error loading image %s\n", inputFile);
        return 1;
    }
    printf("Image loaded: %dx%d (file had %d channels), using %d channels in memory\n",
           width, height, channels_in_file, IN_COMP);

    size_t inBytes  = (size_t)width * height * IN_COMP;
    size_t outBytes = (size_t)width * height * OUT_COMP;

    unsigned char* h_output_gpu = (unsigned char*)malloc(outBytes);
    unsigned char* h_output_cpu = (unsigned char*)malloc(outBytes);
    if (!h_output_gpu || !h_output_cpu) {
        printf("malloc failed\n");
        stbi_image_free(h_input_rgba);
        return 1;
    }

    // ========== Device alloc ==========
    unsigned char* d_input = nullptr;   // RGBA
    unsigned char* d_output = nullptr;  // RGB
    float* d_color_weight = nullptr;

    CHECK(cudaMalloc((void**)&d_input,  inBytes));
    CHECK(cudaMalloc((void**)&d_output, outBytes));
    CHECK(cudaMemcpy(d_input, h_input_rgba, inBytes, cudaMemcpyHostToDevice));

    // ========== Precompute space weights -> constant ==========
    int ds2;
    const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);

    float* h_space_weight = (float*)malloc((size_t)dim_kernel * dim_kernel * sizeof(float));
    if (!h_space_weight) {
        printf("malloc h_space_weight failed\n");
        stbi_image_free(h_input_rgba);
        free(h_output_gpu);
        free(h_output_cpu);
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output));
        return 1;
    }

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            ds2 = (i * i) + (j * j);
            h_space_weight[(i + radius) * dim_kernel + (j + radius)] = expf(-ds2 * inv_2_sigma_s2);
        }
    }
    CHECK(cudaMemcpyToSymbol(space_weight, h_space_weight,
                             (size_t)dim_kernel * dim_kernel * sizeof(float)));

    // ========== Precompute color weights ==========
    const float inv_2_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);

    // dr max = 255*3 = 765 (RGB), quindi serve almeno 766 elementi
    const int CW = 256 * OUT_COMP; // 768, ok
    float* h_color_weight = (float*)malloc((size_t)CW * sizeof(float));
    for (int i = 0; i < CW; i++) {
        h_color_weight[i] = expf((float)(i * i) * -inv_2_sigma_r2);
    }

    CHECK(cudaMalloc((void**)&d_color_weight, (size_t)CW * sizeof(float)));
    CHECK(cudaMemcpy(d_color_weight, h_color_weight, (size_t)CW * sizeof(float), cudaMemcpyHostToDevice));

    // ========== Launch ==========
    int inner_rows = height - (2 * radius);
    if (inner_rows < 0) inner_rows = 0;

    dim3 block((unsigned)block_x, (unsigned)block_y);
    dim3 grid((unsigned)((width + block.x - 1) / block.x),
              (unsigned)((height + block.y - 1) / block.y));
    uchar4 *rgba=d_input;
    bilateral_u8_gray<<<grid, block>>>(rgba, d_output,
                                       width, height,
                                       2 * radius, inner_rows,
                                       radius, d_color_weight, dim_kernel);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // ========== Copy back + save ==========
    CHECK(cudaMemcpy(h_output_gpu, d_output, outBytes, cudaMemcpyDeviceToHost));

    stbi_write_png("risultato.png", width, height, OUT_COMP, h_output_gpu, width * OUT_COMP);
    printf("\nFinito bilateral gpu!!\n\n");

    // ========== CPU reference + verify ==========
    bilateral_u8_gray_cpu(h_input_rgba, h_output_cpu, width, height, radius, sigma_s, sigma_r);

    bool correct = verifyResults(h_output_cpu, h_output_gpu, (int)outBytes, "GPU");
    printf(correct ? "✓ Test PASSATO: GPU e CPU producono lo stesso risultato\n" : "Fallito\n");

    // ========== Cleanup ==========
    stbi_image_free(h_input_rgba);
    free(h_output_gpu);
    free(h_output_cpu);
    free(h_space_weight);
    free(h_color_weight);

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_color_weight));

    return 0;
}
