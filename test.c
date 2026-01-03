/*
Naive bilateral filter (grayscale, uint8).
- in/out: row-major arrays of size width*height
- radius: window radius r (kernel size = (2r+1)^2)
- sigma_s: spatial sigma (pixels)
- sigma_r: range sigma (intensity units, 0..255)

Compile: gcc -O2 bilateral.c -lm
*/
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static inline int clampi(int v, int lo, int hi) {
  return (v < lo) ? lo : (v > hi) ? hi : v;
}

void bilateral_u8_gray(
    const uint8_t *in, uint8_t *out,
    int width, int height,
    int radius, float sigma_s, float sigma_r
) {
  if (!in || !out || width <= 0 || height <= 0 || radius < 0) return;
  if (sigma_s <= 0.0f || sigma_r <= 0.0f) return;

  const float inv_2_sigma_s2 = 1.0f / (2.0f * sigma_s * sigma_s);
  const float inv_2_sigma_r2 = 1.0f / (2.0f * sigma_r * sigma_r);

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

      float res = (wsum > 0.0f) ? (sum / wsum) : center;
      int ir = (int)lroundf(res);
      out[idx0] = (uint8_t)clampi(ir, 0, 255);
    }
  }
}
