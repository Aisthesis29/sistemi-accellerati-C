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

__global__ void memory_alignment(unsigned char *h_input, uchar4* rgba, int width, int height) {
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

__global__ void bilateral_u8_gray(uchar4 *rgba, unsigned char *out, int width, int height, int radius, float *space_weight, float *color_weight, int dim_kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int y0, yn, x0, xn;
    //const float inv_2_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);
        
    if (x < width && y < height) {
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
        int i=0, bordo = 0;
        if(y<=radius || (y+radius)>=height) {
            bordo = 1;
        }
        float w,w_s;
        if(bordo==0){
 for (int dy = y0; dy < y; ++dy) {
            

            for (int dx = x0; dx <= xn; ++dx) {
                int idxUpp = dy*width+dx;
               // if(y==10){
                 //   printf("valore=%d",dim_kernel);
               // }
              
                int idxDown1 = (dy+dim_kernel-1-2*(i))*width+dx;
             int dy_mirror = 2*y - dy; // se dy<y, allora dy_mirror>y 
             int idxDown = dy_mirror * width + dx;
            // if(idx0==DEBUG_IDX){
           //printf( "sono entrato");
             //   printf("down1=%d down=%d\n",idxDown1,idxDown);
                //     }
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
                
              // sum_r=0;
            //sum_g=0;
            //sum_b=0;
            /*
             if(idx0==DEBUG_IDX){
           printf( "SUM_R_G_B=%f %f %f",sum_r,sum_g,sum_b); //valori NON ok
           printf("\nw?=%f",w);
                      printf("\nw_s=%f",w_s);

                      //printf( "val rgU etc U=%d %d %d",val_rU,val_gU,val_bU); valori ok
                      //printf( "val rgD etc D=%d %d %d",val_rD,val_gD,val_bD);valori ok

                }*/
               
            }
             i++;
              //wsum=0;
             
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
        }
        else{
 for (int dy = y0; dy <= yn; ++dy) {
            for (int dx = x0; dx <= xn; ++dx) {
                int idx = dy * width + dx;

             /*   if(y==10&&x==2){
                    printf("    CIAO DA BORDO=%d",dim_kernel);
                }*/
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
      
       
       
       /* if(idx0==DEBUG_IDX){
           printf("gPU idx=%d\nin   = R:%d G:%d B:%d\n"
                  "\ntmp  = R:%d G:%d B:%d\n",
                    idx0,
                    
                    rgba[idx0].x,
                    rgba[idx0].y,
                    rgba[idx0].z,
                    out[idx0*3],
                    out[idx0*3+1],
                    out[idx0*3+2]);
        }*/
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
            if (errors < 4) {
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
                //int yy = clampi_cpu(y + dy, 0, height - 1);

                for (int dx = x0; dx <= xn; ++dx) {
                    //int xx = clampi_cpu(x + dx, 0, width - 1);
                    //int idx = yy * width + xx;
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

            /*if (idx0 > (height * width) / 2 && out[idx0 * 3] >= 200&&peso>=2 ) {
                printf("idx0 = %d, peso=%f\n", idx0,peso);
            
                printf("original (h_input): R=%u G=%u B=%u\n",
                                                              h_input[idx0 * 3],
                                                              h_input[idx0 * 3 + 1],
                                                              h_input[idx0 * 3 + 2]);

                printf("new (out): R=%u G=%u B=%u\n",
                                                    out[idx0 * 3],
                                                    out[idx0 * 3 + 1],
                                                    out[idx0 * 3 + 2]);
            }*/

        }
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
    /*const char* inputFile = argv[1];
    int radius = atoi(argv[2]);
    int sigma_s = atoi(argv[3]);
    int  sigma_r = atoi(argv[4]);*/

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

    dim3 block(blockSize, blockSize);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    // ========== Operazione reali ==========
    memory_alignment<<<grid, block>>>(d_input, rgba, width, height);
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
            //printf("dr: %d, %f\n", i, color_weight[i]);
    }
    float *d_color_weight;

    CHECK(cudaMalloc((void**)&d_color_weight, sizeof(float)*cn*256));
    CHECK(cudaMemcpy(d_color_weight, color_weight, sizeof(float)*cn*256, cudaMemcpyHostToDevice));
    bilateral_u8_gray<<<grid, block>>>(rgba, d_output, width, height, radius, d_space_weight, d_color_weight, dim_kernel);

    // ========== Salvataggio immagini ==========
    CHECK(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));
    //CHECK(cudaMemcpy(d_output, h_output, imageSize, cudaMemcpyDeviceToHost));
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