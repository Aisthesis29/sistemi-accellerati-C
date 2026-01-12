#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

// Include STB image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static inline int clampi(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

/*void grayscale_cpu(unsigned char* input, unsigned char* output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            
            unsigned char r = input[idx + 0];
            unsigned char g = input[idx + 1];
            unsigned char b = input[idx + 2];
            
            float sum = fmaf(0.299f, r, fmaf(0.587f, g, 0.114f * b));
            unsigned char gray = (unsigned char)(sum);
            
            int out_idx = y * width + x;
            output[out_idx] = gray;
        }
    }
}*/

void bilateral_u8_gray(unsigned char *h_input, unsigned char *out, int width, int height, int radius, int sigma_s, int sigma_r) {
    if(!out || ! h_input) {
        printf("errore nel caricamento di una matrice\n");
        printf("out=[%d]\t", out);
        printf("h_input=[%d]", h_input);
    }
    printf("width=%d, height=%d, radius=%d\n", width, height, radius);
    printf("sigma_s=%d, sigma_r=%d\n", sigma_s, sigma_r);
    if (!out || width <= 0 || height <= 0 || radius < 0) 
        return;
    if (sigma_s <= 0 || sigma_r <= 0) 
        return;

    const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
    const float inv_2_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);
    float somma=0, num=0;
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

            for (int dy = -radius; dy <= radius; ++dy) {
                int yy = clampi(y + dy, 0, height - 1);

                for (int dx = -radius; dx <= radius; ++dx) {
                    int xx = clampi(x + dx, 0, width - 1);
                    int idx = yy * width + xx;

                    int val_r = (int)h_input[idx*3];
                    int val_g = (int)h_input[idx*3+1];
                    int val_b = (int)h_input[idx*3+2];

                    int ds2 = (dx * dx + dy * dy);
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
            /*
            float res = (wsum > 0.0f) ? (sum / wsum) : center;
            int ir = (int)lroundf(res);
            out[idx0] = (unsigned char)clampi(ir, 0, 255);
            */
            float inv_Wsum = 1.f/wsum;
            out[idx0*3] = (unsigned char)(sum_r*inv_Wsum+0.5);
            out[idx0*3+1] = (unsigned char)(sum_g*inv_Wsum+0.5);
            out[idx0*3+2] = (unsigned char)(sum_b*inv_Wsum+0.5);

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
    //printf("valore medio peso %f", somma/num);
}

int main(int argc, char **argv) {
    printf("Compilo");
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
    
    // ========== Caricamento immagine ==========
    int width, height, channels;
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels, 0);
    if (!h_input) {
        printf("Error loading image %s\n", inputFile);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

    //OPERAZIONI BELLE
    // ========== Allocazione memoria ==========
    int imageSize = width * height * channels;
    int grayscale = imageSize/channels;
    unsigned char* h_output = (unsigned char*)malloc(imageSize);

    /*unsigned char *in = (unsigned char*)malloc(grayscale);
    grayscale_cpu(h_input, in, width, height);
    printf("Finito greyscale\n");
    stbi_write_png("grayscale.png", width, height, 1, in, width);*/
    bilateral_u8_gray(h_input, h_output, width, height, radius, sigma_s, sigma_r);
    
    for(int i=0; i<150; i++) {
        printf("id: %d, CPU: %d\n", i, h_output[i]);
    }
    // ========== Salvataggio immagini ==========
    stbi_write_png("risultato.png", width, height, channels, h_output, width * channels);

    free(h_input);
    free(h_output);
    printf("\n\n");
    return 0;
}