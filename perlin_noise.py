import matplotlib.pyplot as plt
import numpy as np
import random as rand
import math

def clamp(x, lowerlimit, upperlimit):
  if x < lowerlimit:
    x = lowerlimit
  if x > upperlimit:
    x = upperlimit
  return x


def lerp(a0, a1, w):
    w = w*w*w*(w*(w*6-15)+10)
    return (1 - w)*a0 + w*a1


# Compute Perlin noise at coordinates x, y
def perlin(x, y):
    # Determine grid cell coordinates
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1

    # Determine interpolation weights
    # Could also use higher order polynomial/s-curve here
    sx = x - x0
    sy = y - y0

    tl = grad[x0][y0]
    tr = grad[x1][y0]
    bl = grad[x0][y1]
    br = grad[x1][y1]

    # Interpolate between grid point gradients
    n0 = tl[0] * sx + tl[1] * sy
    n1 = tr[0] * (sx - 1) + tr[1] * sy
    ix0 = lerp(n0, n1, sx)

    n0 = bl[0] * sx + bl[1] * (sy - 1)
    n1 = br[0] * (sx - 1) + br[1] * (sy - 1)
    ix1 = lerp(n0, n1, sx)

    value = lerp(ix0, ix1, sy)
    return value


width = 1000
height = 1000
cell_count_x = 5
cell_count_y = 5
grad = np.zeros((cell_count_x + 1, cell_count_y + 1, 2))
for yi in range(cell_count_y + 1):
    for xi in range(cell_count_x + 1):
        x = rand.random() * 2 - 1
        y = rand.random() * 2 - 1
        mag = math.sqrt(x * x + y * y)
        grad[yi][xi][0] = x / mag
        grad[yi][xi][1] = y / mag

img = np.zeros((width, height))
for yi in range(height):
    for xi in range(width):
        y = yi / height
        x = xi / width
        p = (perlin(x * cell_count_x, y * cell_count_y) + 1) / 2
        img[yi][xi] = 1 if p > .5 else (0.5 if p > .4 else 0)

plt.imshow(img)
plt.show()