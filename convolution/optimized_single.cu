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

__constant__ float space_weight[DIM*DIM];

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
            if (grave_errors<20 && diff>1) {
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
        idx0 = fmaf(y,width,x);//y * width + x;

        uchar4 pixel = rgba[idx0];
        int center_r = (int)pixel.x;
        int center_g = (int)pixel.y;
        int center_b = (int)pixel.z;
        
        int i=0;
        float w,w_s;
        for (int dy = y-radius; dy < y; ++dy) {
            for (int dx = x0; dx <= xn; ++dx) {
                int idxUpp = fmaf(dy,width,dx);//dy*width+dx;
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
                int dr  = abs(val_rU - center_r)+abs(val_gU - center_g)+abs(val_bU - center_b);
                w = w_s * color_weight[dr];
                wsum += w;
                sum_r = fmaf(w, val_rU, sum_r);  //sum_r += w * val_r;
                sum_g = fmaf(w, val_gU, sum_g);  //sum_g += w * val_g;
                sum_b = fmaf(w, val_bU, sum_b);  //sum_b += w * val_b;

                int drD  = abs(val_rD - center_r)+abs(val_gD - center_g)+abs(val_bD - center_b);
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

            int dr  = abs(val_r - center_r)+abs(val_g - center_g)+abs(val_b - center_b);
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

                int dr  = abs(val_r - center_r)+abs(val_g - center_g)+abs(val_b - center_b);
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

                int dr  = abs(val_r - center_r)+abs(val_g - center_g)+abs(val_b - center_b);
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



void bilateral_u8_gray_cpu(unsigned char *h_input, unsigned char *out, int width, int height, int radius, int sigma_s, int sigma_r) {
    if(!out || ! h_input) {
        printf("errore nel caricamento di una matrice\n");
        printf("out=[%d]\t", out);
        printf("h_input=[%d]", h_input);
    }
    if (!out || width <= 0 || height <= 0 || radius < 0) 
        return;
    if (sigma_s <= 0 || sigma_r <= 0) 
        return;

    int y0, yn, x0, xn;
    const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
    const float inv_2_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx0 = y * width + x;
            int center_r = (int)h_input[idx0*3];
            int center_g = (int)h_input[idx0*3+1];
            int center_b = (int)h_input[idx0*3+2];

            float wsum = 0.0f;
            float sum_r = 0.0f;
            float sum_g = 0.0f;
            float sum_b = 0.0f;

            y0 = max_cpu(y-radius, 0);
            yn = min_cpu(y+radius, height-1);
            x0 = max_cpu(x-radius, 0);
            xn = min_cpu(x+radius, width-1);
            for (int dy = y0; dy <= yn; ++dy) {
                for (int dx = x0; dx <= xn; ++dx) {
                    int idx = dy * width + dx;

                    int val_r = (int)h_input[idx*3];
                    int val_g = (int)h_input[idx*3+1];
                    int val_b = (int)h_input[idx*3+2];

                    int ds2 = (dx-x) * (dx-x) + (dy-y) * (dy-y);
                    int dr  = abs(val_r - center_r)+abs(val_g - center_g)+abs(val_b - center_b);
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
            float inv_Wsum = 1.f/wsum;
            out[idx0*3] = (unsigned char)(sum_r*inv_Wsum+0.5f);
            out[idx0*3+1] = (unsigned char)(sum_g*inv_Wsum+0.5f);
            out[idx0*3+2] = (unsigned char)(sum_b*inv_Wsum+0.5f);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 7) {
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
    int block_x = atoi(argv[5]);
    int block_y = atoi(argv[6]);
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
    
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels, 0);
    if (!h_input) {
        printf("Error loading image %s\n", inputFile);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

    //OPERAZIONI BELLE GPU
    // ========== Allocazione memoria ==========
    int imageSize = width * height * channels;
    unsigned char* h_output = (unsigned char*)malloc(imageSize);

    // ========== Allocazione device ==========
    uchar4 *rgba;
    unsigned char *d_output, *d_input;
    CHECK(cudaMalloc((void**)&d_input, imageSize));  //matchata al 4 della stbi_load in riga 269
    CHECK(cudaMalloc((void**)&rgba, width*height*4));
    CHECK(cudaMalloc((void**)&d_output, imageSize));

    CHECK(cudaMemcpy(d_input, h_input,  imageSize, cudaMemcpyHostToDevice)); //match anche qui

    dim3 block(block_x, block_y);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    // ========== Operazioni reali ==========
    memory_bella<<<grid, block>>>(d_input, rgba, width, height);
    int ds2;
    const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
    //space weight
    float *h_space_weight = (float*)malloc(dim_kernel*dim_kernel*sizeof(float));
    for(int i=-radius; i<=radius; i++) {
        for(int j=-radius; j<=radius; j++) {
            ds2 = (i*i)+(j*j);
            h_space_weight[(i+radius)*dim_kernel+j+radius] = expf(-ds2 * inv_2_sigma_s2);
        }
    }
    cudaMemcpyToSymbol(space_weight, h_space_weight, dim_kernel*dim_kernel*sizeof(float));

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

    int inner_rows=height-(2*radius);
    dim3 grid_inner((width + block.x - 1) / block.x, (inner_rows + block.y - 1) / block.y);

    bilateral_u8_gray<<<grid, block>>>(rgba, d_output, width,height,2*radius,inner_rows, radius, d_color_weight, dim_kernel);
    /*int rows =radius;
    dim3 grid_row((width + block.x - 1) / block.x,
                (2 * rows  + block.y - 1) / block.y);   //2* per considerare che ora fa sopra e sotto (2* thread necessari)
    bilateral_u8_gray_unopt_ybase<<<grid_row, block>>>(
        rgba, d_output,                 // base pointers (immagine intera)
        width, height,                  // DIMENSIONI REALI
        rows+inner_rows, 2*radius,                           // y_base=0, rows=1  -> solo prima riga
        radius,
        d_color_weight,
        dim_kernel
    );*/
    // ========== Salvataggio immagini ==========
    
    CHECK(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    stbi_write_png("risultato.png", width, height, channels, h_output, width * channels);
    printf("\nFinito bilateral gpu!!\n\n");

    //Parte CPU + controllo
    unsigned char* h_output_cpu = (unsigned char*)malloc(imageSize);
    bilateral_u8_gray_cpu(h_input, h_output_cpu, width, height, radius, sigma_s, sigma_r);
    
    bool correct = verifyResults(h_output_cpu, h_output, imageSize, "GPU");
    if (correct) {
        printf("✓ Test PASSATO: GPU e CPU producono lo stesso risultato\n");
    } else {
        printf("Fallito\n");
    }

    free(h_input);
    free(h_output);
    free(h_output_cpu);
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(rgba));

    printf("\n\n");
    return 0;
} 