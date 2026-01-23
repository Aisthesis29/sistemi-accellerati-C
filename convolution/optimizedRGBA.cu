#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
//test
// Include STB image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define DEBUG_IDX 7680


/*__device__ int clampi(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}
static inline int clampi_cpu(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}*/

/*__device__ int min(int v1, int v2) {
  return (v1 > v2) ? v2 : v1;
}
__device__ int max(int v1, int v2) {
  return (v1 < v2) ? v2 : v1;
}*/

static inline int min_cpu(int v1, int v2) {
  return (v1 > v2) ? v2 : v1;
}
static inline int max_cpu(int v1, int v2) {
  return (v1 < v2) ? v2 : v1;
}

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

__global__ void memory_bella(unsigned char *h_input, uchar4* rgba, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel = y * width + x; 
        int idx = pixel *3;
        
        unsigned char r = h_input[idx];
        unsigned char g = h_input[idx+1];
        unsigned char b = h_input[idx+2];
        rgba[pixel] = make_uchar4(r, g, b, 255);
    }
}
// Funzione per verificare correttezza
bool verifyResults(unsigned char* cpu_result, unsigned char* gpu_result, int size, const char* label)
{
    int errors = 0;
    int grave_errors=0;
    for (int i = 0; i < size; i++) {
        // Tolleriamo differenze di ±1 dovute ad arrotondamenti
        int diff = abs((int)cpu_result[i] - (int)gpu_result[i]);
        if (diff >= 1) {
            errors++;
            if(diff>=2){
                grave_errors++;
            }
            if (errors < 200) {
                printf("Mismatch at index %d: CPU=%d, %s=%d (diff=%d)\n", 
                        i, cpu_result[i], label, gpu_result[i], diff);
            }
        }
    }
    
    if (errors > 0) {
        printf("Total errors: %d (grave: %d)\n", errors, grave_errors);
        return false;
    }
    return true;
}

// CPU reference (bilateral)
void bilateral_u8_gray_cpu(unsigned char* input, unsigned char* output, int width, int height, int radius, int sigma_s, int sigma_r) {
    const int dim_kernel = 2 * radius + 1;
    const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
    const float inv_2_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);

    // precompute space weights
    float* space_weight = (float*)malloc(dim_kernel * dim_kernel * sizeof(float));
    for(int j=-radius; j<=radius; j++) {
        for(int i=-radius; i<=radius; i++) {
            int ds2 = i*i + j*j;
            space_weight[(j+radius)*dim_kernel + (i+radius)] = expf(-ds2 * inv_2_sigma_s2);
        }
    }

    // precompute color weights (cn=3)
    const int cn = 3;
    float* color_weight = (float*)malloc(cn * 256 * sizeof(float));
    for(int i=0; i<256*cn; i++) {
        color_weight[i] = expf(i*i * -inv_2_sigma_r2);
    }

    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            int idx0 = y*width + x;
            int base0 = idx0*cn;

            int center_r = input[base0 + 0];
            int center_g = input[base0 + 1];
            int center_b = input[base0 + 2];

            float wsum=0.f, sum_r=0.f, sum_g=0.f, sum_b=0.f;

            int y0 = max_cpu(y-radius, 0);
            int yn = min_cpu(y+radius, height-1);
            int x0 = max_cpu(x-radius, 0);
            int xn = min_cpu(x+radius, width-1);

            for(int dy=y0; dy<=yn; dy++) {
                for(int dx=x0; dx<=xn; dx++) {
                    int idx = dy*width + dx;
                    int base = idx*cn;

                    int val_r = input[base + 0];
                    int val_g = input[base + 1];
                    int val_b = input[base + 2];

                    int dr = abs(val_r-center_r) + abs(val_g-center_g) + abs(val_b-center_b);
                    float w = space_weight[(dy - y + radius)*dim_kernel + (dx - x + radius)] * color_weight[dr];

                    wsum += w;
                    sum_r += w * val_r;
                    sum_g += w * val_g;
                    sum_b += w * val_b;
                }
            }

            float invW = 1.f/wsum;
            output[base0 + 0] = (unsigned char)(sum_r*invW + 0.5f);
            output[base0 + 1] = (unsigned char)(sum_g*invW + 0.5f);
            output[base0 + 2] = (unsigned char)(sum_b*invW + 0.5f);
        }
    }

    free(space_weight);
    free(color_weight);
}

__global__ void bilateral_u8_gray_unopt_ybase(
    const uchar4 *rgba, uchar4 *out,
    int width, int height,
    int y_base, int rows,        // regione: [y_base, y_base+rows)
    int radius, const float *space_weight, const float *color_weight, int dim_kernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_local = blockIdx.y * blockDim.y + threadIdx.y;

    int y0, yn, x0, xn;

    if (x >= width || y_local >= rows) return;

    if (x < width && y_local < height) {
        int y = y_base + y_local;               // y globale

        const int idx0 = y * width + x;

        uchar4 pixel = rgba[idx0];
        int center_r = (int)pixel.x;
        int center_g = (int)pixel.y;
        int center_b = (int)pixel.z;

        float wsum = 0.0f;
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;

        y0 = max(y-radius, 0);
        yn = min(y+radius, height-1);
        x0 = max(x-radius, 0);
        xn = min(x+radius, width-1);

        for (int dy = y0; dy <= yn; ++dy) {
            for (int dx = x0; dx <= xn; ++dx) {
                int idx = dy * width + dx;

                uchar4 val = rgba[idx];
                int val_r = (int)val.x;
                int val_g = (int)val.y;
                int val_b = (int)val.z;

                int dr  = abs(val_r - center_r)+abs(val_g - center_g)+abs(val_b - center_b);
                float w = space_weight[(dx-x+radius)*dim_kernel+(dy-y+radius)] * color_weight[dr];

                wsum += w;
                sum_r = fmaf(w, val_r, sum_r);
                sum_g = fmaf(w, val_g, sum_g);
                sum_b = fmaf(w, val_b, sum_b);
            }
        }

        float inv_Wsum = 1.f / wsum;
        unsigned char rr = (unsigned char)(sum_r * inv_Wsum + 0.5f);
        unsigned char gg = (unsigned char)(sum_g * inv_Wsum + 0.5f);
        unsigned char bb = (unsigned char)(sum_b * inv_Wsum + 0.5f);
        out[idx0] = make_uchar4(rr, gg, bb, 255);
    }
}

__global__ void bilateral_u8_gray(uchar4 *rgba, uchar4 *out, int width, int height, int y_base, int rows, int radius, float *space_weight, float *color_weight, int dim_kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_local = blockIdx.y * blockDim.y + threadIdx.y;

    int y0, yn, x0, xn;
        
    if (x >= width || y_local >= rows) return;

    if (x < width && y_local < height) {
        int y = y_base + y_local;
        const int idx0 = y * width + x;

        uchar4 pixel = rgba[idx0];
        int center_r = (int)pixel.x;
        int center_g = (int)pixel.y;
        int center_b = (int)pixel.z;

        float wsum = 0.0f;
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;

        y0 = max(y-radius, 0);
        yn = min(y+radius, height-1);
        x0 = max(x-radius, 0);
        xn = min(x+radius, width-1);

        for (int dy = y0; dy <= yn; ++dy) {
            for (int dx = x0; dx <= xn; ++dx) {
                int idx = dy * width + dx;

                uchar4 val = rgba[idx];
                int val_r = (int)val.x;
                int val_g = (int)val.y;
                int val_b = (int)val.z;

                int dr  = abs(val_r - center_r)+abs(val_g - center_g)+abs(val_b - center_b);
                float w = space_weight[(dx-x+radius)*dim_kernel+(dy-y+radius)] * color_weight[dr];

                wsum += w;
                sum_r = fmaf(w, val_r, sum_r);
                sum_g = fmaf(w, val_g, sum_g);
                sum_b = fmaf(w, val_b, sum_b);
            }
        }

        float inv_Wsum = 1.f/wsum;
        unsigned char rr = (unsigned char)(sum_r * inv_Wsum + 0.5f);
        unsigned char gg = (unsigned char)(sum_g * inv_Wsum + 0.5f);
        unsigned char bb = (unsigned char)(sum_b * inv_Wsum + 0.5f);
        out[idx0] = make_uchar4(rr, gg, bb, 255);
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: %s <frame_pattern> <kernel_size> <sigma_s> <sigma_r>\n", argv[0]);
        printf("kernel_size: dimensione matrice di convoluzione, intero dispari\n");
        printf("sigma_s: parametro relativo allo spazio, intero positivo\n");
        printf("sigma_r: parametro relativo all'intensità luminosa, intero positivo\n");
        return 1;
    }
    const char* inputFile = argv[1];
    int dim_kernel = atoi(argv[2]);
    int sigma_s = atoi(argv[3]);
    int  sigma_r = atoi(argv[4]);
    if(dim_kernel%2==0 || dim_kernel<1) {
        printf("kernel_size: valore non valido. Deve essere un intero positivo dispari\n");
        return 2;
    }
    if(sigma_s<=0) {
        printf("sigma_s: valore non valido. Deve essere un intero positivo\n");
        return 2;
    }
    if(sigma_r<=0) {
        printf("sigma_r: valore non valido. Deve essere un intero positivo\n");
        return 2;
    }
    int radius = dim_kernel/2;

    int blockSize = 16;

    // ========== Caricamento immagine ==========
    int width, height, channels;
    
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels, 3);
    channels = 3;
    if (!h_input) {
        printf("Error loading image %s\n", inputFile);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

    //OPERAZIONI BELLE GPU
    // ========== Allocazione memoria ==========
    int imageSize = width * height * channels;
    uchar4* h_output_rgba = (uchar4*)malloc((size_t)width * height * sizeof(uchar4));
    unsigned char* h_output_rgb = (unsigned char*)malloc((size_t)width * height * 3);

    // ========== Allocazione device ==========
    uchar4 *rgba;
    unsigned char *d_input;
    uchar4 *d_output_rgba;
    CHECK(cudaMalloc((void**)&d_input, imageSize));
    CHECK(cudaMalloc((void**)&rgba, (size_t)width * height * sizeof(uchar4)));
    CHECK(cudaMalloc((void**)&d_output_rgba, (size_t)width * height * sizeof(uchar4)));

    CHECK(cudaMemcpy(d_input, h_input,  imageSize, cudaMemcpyHostToDevice));

    dim3 block(blockSize, blockSize);
    dim3 grid((width + block.x - 1) / block.x, (height) + block.y - 1) / block.y);
    // ========== Operazioni reali ==========
    memory_bella<<<grid, block>>>(d_input, rgba, width, height);
    int ds2;
    float space_weight[dim_kernel][dim_kernel];
    const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
    //space weight
    for(int i=-radius; i<=radius; i++) {
        for(int j=-radius; j<=radius; j++) {
            ds2 = (i*i)+(j*j);
            space_weight[i+radius][j+radius] = expf(-ds2 * inv_2_sigma_s2);
        }
    }
    int convSize=(dim_kernel*dim_kernel)*sizeof(float);
    float *d_space_weight;
    CHECK(cudaMalloc((void**)&d_space_weight, convSize));
    CHECK(cudaMemcpy(d_space_weight, space_weight, convSize, cudaMemcpyHostToDevice));

    //color weight
    const float inv_2_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);
    const int cn=channels;
    float color_weight[cn*256];
    for( int i = 0; i < 256 * cn; i++ ){
            color_weight[i] = expf(i * i * -inv_2_sigma_r2);
    }
    float *d_color_weight;

    CHECK(cudaMalloc((void**)&d_color_weight, sizeof(float)*cn*256));
    CHECK(cudaMemcpy(d_color_weight, color_weight, sizeof(float)*cn*256, cudaMemcpyHostToDevice));

    uchar4 *rgba_inner=rgba+(width*dim_kernel);

    uchar4 *rgba_border1=rgba;

    uchar4 *rgba_inner_end= rgba+(width*height)-(width*dim_kernel);

    //######################second try
    uchar4 *first_row_end=rgba+width*dim_kernel;
    uchar4 *last_row_start=rgba+(width*height)-(width);

    uchar4 *out_first = d_output_rgba;

    int inner_rows=height-(dim_kernel*2);
    dim3 grid_inner((width + block.x - 1) / block.x, (inner_rows + block.y - 1) / block.y);

    bilateral_u8_gray<<<grid_inner, block>>>(rgba, out_first, width, height, dim_kernel, inner_rows, radius, d_space_weight, d_color_weight, dim_kernel);

    int rows =dim_kernel;
    dim3 grid_row((width + block.x - 1) / block.x,
                  (rows  + block.y - 1) / block.y);
    bilateral_u8_gray_unopt_ybase<<<grid_row, block>>>(
        rgba, out_first,
        width, height,
        0, rows,
        radius,
        d_space_weight, d_color_weight,
        dim_kernel
    );

    //last row
    bilateral_u8_gray_unopt_ybase<<<grid_row, block>>>(
        rgba, out_first,
        width, height,
        rows+inner_rows, rows,
        radius,
        d_space_weight, d_color_weight,
        dim_kernel
    );

    // ========== Salvataggio immagini ==========
    
    CHECK(cudaMemcpy(h_output_rgba, d_output_rgba, (size_t)width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    stbi_write_png("risultato.png", width, height, 4, h_output_rgba, width * 4);

    // Converti RGBA -> RGB per confronto con CPU (che lavora a 3 canali)
    for (int i = 0; i < width * height; ++i) {
        h_output_rgb[3*i + 0] = h_output_rgba[i].x;
        h_output_rgb[3*i + 1] = h_output_rgba[i].y;
        h_output_rgb[3*i + 2] = h_output_rgba[i].z;
    }

    printf("\nFinito bilateral gpu!!\n\n");

    //Parte CPU + controllo
    unsigned char* h_output_cpu = (unsigned char*)malloc(imageSize);
    bilateral_u8_gray_cpu(h_input, h_output_cpu, width, height, radius, sigma_s, sigma_r);
    
    bool correct = verifyResults(h_output_cpu, h_output_rgb, imageSize, "GPU");
    if (correct) {
        printf("✓ Test PASSATO: GPU e CPU producono lo stesso risultato\n");
    } else {
        printf("Fallito\n");
    }

    free(h_input);
    free(h_output_rgba);
    free(h_output_rgb);
    free(h_output_cpu);
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output_rgba));
    CHECK(cudaFree(rgba));

    printf("\n\n");
    return 0;
} 
