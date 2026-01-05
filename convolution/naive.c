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

static inline int clampi(int v, int lo, int hi) {
  return (v < lo) ? lo : (v > hi) ? hi : v;
}

// Versione CPU per confronto
// Versione CPU per confronto
void grayscale_cpu(unsigned char* input, unsigned char* output, int width, int height) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * 3;
                
                unsigned char r = input[idx + 0];
                unsigned char g = input[idx + 1];
                unsigned char b = input[idx + 2];
                
                unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
                
                int out_idx = y * width + x;
                output[out_idx] = gray;
            }
        }
}
/*__global__ void grayscale_cpu(unsigned char* input, unsigned char* output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = y * width + +;
    unsigned char r = input[idx + 0];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 2];

    unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);

    int out_idx = y * width + x;
    output[out_idx] = gray;
  }
}*/

void bilateral_u8_gray(unsigned char *h_input, unsigned char *in, unsigned char *out, int width, int height, int radius, int sigma_s, int sigma_r) {
  if(!in || !out || ! h_input) {
    printf("errore nel caricamento di una matrice\n");
    printf("in=[%d]\t", in);
    printf("out=[%d]\t", out);
    printf("h_input=[%d]", h_input);
  }
  printf("width=%d, height=%d, radius=%d\n", width, height, radius);
  printf("sigma_s=%d, sigma_r=%d\n", sigma_s, sigma_r);
  if (!in || !out || width <= 0 || height <= 0 || radius < 0) return;
  if (sigma_s <= 0 || sigma_r <= 0) return;

  const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
  const float inv_2_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);
float somma=0, num=0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int idx0 = y * width + x;
      const float center = (float)in[idx0];

      float wsum = 0.0f;
      float sum  = 0.0f;

      for (int dy = -radius; dy <= radius; ++dy) {
        const int yy = clampi(y + dy, 0, height - 1);

        for (int dx = -radius; dx <= radius; ++dx) {
          const int xx = clampi(x + dx, 0, width - 1);
          const int idx = yy * width + xx;

          const float val = (float)in[idx];

          const float ds2 = (float)(dx * dx + dy * dy);
          const float dr  = val - center;
          const float dr2 = dr * dr;

          const float w_s = expf(-ds2 * inv_2_sigma_s2);
          const float w_r = expf(-dr2 * inv_2_sigma_r2);
          const float w   = w_s * w_r;

          wsum += w;
          sum  += w * val;
        }
      }
      /*
      float res = (wsum > 0.0f) ? (sum / wsum) : center;
      int ir = (int)lroundf(res);
      out[idx0] = (unsigned char)clampi(ir, 0, 255);
      */
      float peso = (sum/wsum)/center;
      if(peso>2){
        peso=2;
      } else {
        somma+=peso;
        num++;
      }
      int tmp = (int)lroundf(h_input[idx0*3]*peso);
      out[idx0*3] = (unsigned char)clampi(tmp, 0, 255);
      tmp = (int)lroundf(h_input[idx0*3+1]*peso);
        out[idx0*3+1] = (unsigned char)clampi(tmp, 0, 255);
        tmp = (int)lroundf(h_input[idx0*3+2]*peso);
        out[idx0*3+2] = (unsigned char)clampi(tmp, 0, 255);


       if (idx0 > (height * width) / 2 && out[idx0 * 3] >= 200&&peso>=2 ) {
    printf("idx0 = %d, peso=%f\n", idx0,peso);
      
    printf("original (h_input): R=%u G=%u B=%u\n",
           h_input[idx0 * 3],
           h_input[idx0 * 3 + 1],
           h_input[idx0 * 3 + 2]);

    printf("new (out):          R=%u G=%u B=%u\n",
           out[idx0 * 3],
           out[idx0 * 3 + 1],
           out[idx0 * 3 + 2]);
}

    }
  }
  printf("valore medio peso %f", somma/num);
}

int main(int argc, char **argv) {
        
  const char* inputFile = argv[1];
  int radius = atoi(argv[2]);
  int sigma_s = atoi(argv[3]);
  int  sigma_r = atoi(argv[4]);
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

  unsigned char *in = (unsigned char*)malloc(grayscale);
  grayscale_cpu(h_input, in, width, height);
  printf("Finito greyscale\n");
  stbi_write_png("grayscale.png", width, height, 1, in, width);
  bilateral_u8_gray(h_input, in, h_output, width, height, radius, sigma_s, sigma_r);

  // ========== Salvataggio immagini ==========
  stbi_write_png("risultato.png", width, height, channels, h_output, width * channels);

  free(h_input);
  free(h_output);
  free(in);
  printf("\n\n");
  return 0;
}