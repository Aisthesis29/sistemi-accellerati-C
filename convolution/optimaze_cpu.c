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
#define DEBUG_IDX 10

static inline int min(int v1, int v2) {
  return (v1 > v2) ? v2 : v1;
}
static inline int max(int v1, int v2) {
  return (v1 < v2) ? v2 : v1;
}

void bilateral_u8_gray(unsigned char *h_input, unsigned char *out, int width, int height, int radius, float* space_weight, int sigma_r, int dim_kernel) {
    if(!out || ! h_input) {
        printf("errore nel caricamento di una matrice\n");
        printf("out=[%d]\t", out);
        printf("h_input=[%d]", h_input);
        return;
    }
    if(width <= 0 || height <= 0 || radius < 0) {
        printf("width=%d, height=%d, radius=%d\n", width, height, radius);
    }

    int y0, yn, x0, xn;
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

            y0 = max(y-radius, 0);
            yn = min(y+radius, height-1);
            x0 = max(x-radius, 0);
            xn = min(x+radius, width-1);
            /*if(idx0==DEBUG_IDX/3) {
                printf("optimaze variables\n");
                printf("y: %d, x: %d, radius: %d\n", y, x, radius);
                printf("x0: %d, xn: %d, y0: %d, yn: %d\n", x0, xn, y0, yn);
            }*/
            for (int dy = y0; dy <= yn; ++dy) {
                for (int dx = x0; dx <= xn; ++dx) {
                    int idx = dy * width + dx;


                    int val_r = (int)h_input[idx*3];
                    int val_g = (int)h_input[idx*3+1];
                    int val_b = (int)h_input[idx*3+2];

                    int dr  = abs(val_r - center_r)+abs(val_g - center_g)+abs(val_b - center_b);
                    int dr2 = dr * dr;

                    /*if(idx0==DEBUG_IDX/3) {
                        printf("center:\t%d %d %d\n", center_r, center_g, center_b);
                        printf("val:\t%d %d %d\n", val_r, val_g, val_b);
                        printf("cmp dr: %d %d %d\n", abs(val_r - center_r), abs(val_g - center_g), abs(val_b - center_b));
                        printf("dr: %d, dr2: %d, dx: %d, dy: %d\n", dr, dr2, dx-x, dy-y);
                    }*/

                    float w_r = expf(-dr2 * inv_2_sigma_r2);
                    float w = space_weight[(dx-x+radius)*dim_kernel+(dy-y+radius)] * w_r;

                    wsum += w;
                    sum_r += w * val_r;
                    sum_g += w * val_g;
                    sum_b += w * val_b;

                    /*if(idx0==DEBUG_IDX/3) {
                        printf("w_s: %f, w_r: %f\n", space_weight[(dx-x+radius)*dim_kernel+(dy-y+radius)], w_r);
                        printf("w: %f, wsum: %f, sum: %f %f %f\n", w, wsum, sum_r, sum_g, sum_b);
                    }*/
                }
            }
            float inv_Wsum = 1.f/wsum;
            out[idx0*3] = (unsigned char)(sum_r*inv_Wsum+0.5);
            out[idx0*3+1] = (unsigned char)(sum_g*inv_Wsum+0.5);
            out[idx0*3+2] = (unsigned char)(sum_b*inv_Wsum+0.5);

            /*if(idx0==DEBUG_IDX/3){
                printf("\noptimaze idx=%d\tinv_Wsum = %.9f\nsum = R:%d G:%d B:%d\n"
                        "sum*inv_Wsum+0.5 = R:%.9f G:%.9f B:%.9f\ntmp = R:%d G:%d B:%d\n\n",
                        idx0,
                        inv_Wsum,
                        sum_r,
                        sum_g,
                        sum_b,
                        sum_r*inv_Wsum+0.5,
                        sum_g*inv_Wsum+0.5,
                        sum_b*inv_Wsum+0.5,
                        out[idx0*3], out[idx0*3+1], out[idx0*3+2]);
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
        printf("kerqwnel_size: valore non valido. Deve essere un intero positivo dispari\n");
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
    unsigned char* h_output_naive = (unsigned char*)malloc(imageSize);

    int ds2;
    float space_weight[dim_kernel][dim_kernel];
    const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
    for(int i=-radius; i<=radius; i++) {
        for(int j=-radius; j<=radius; j++) {
            ds2 = (i*i)+(j*j);
            space_weight[i+radius][j+radius] = expf(-ds2 * inv_2_sigma_s2);
        }
    }
    bilateral_u8_gray(h_input, h_output, width, height, radius, space_weight, sigma_r, dim_kernel);
    
    // ========== Salvataggio immagini ==========
    stbi_write_png("risultato.png", width, height, channels, h_output, width * channels);

    free(h_input);
    free(h_output);
    printf("\n\n");
    return 0;
}