1) Choose an 8-bit colored image and convert it to a grayscale image. Explain your
approach and show the input and output image.

**Approach:**
For each pixel, read `R,G,B` (8-bit each) and compute luminescence using the linear approximation:
$$
Y = 0.299R  + 0.587G + 0.114B
$$

```go
func toGray(src image.Image) *image.Gray {
    bounds := src.Bounds()
    dst := image.NewGray(bounds)

    // convert each pixel to grayscale using luminance formula
    // Y' = 0.299 R + 0.587 G + 0.114 B
    // where R, G, and B are in the range [0, 255]
    // and Y' is the resulting luminance value
    // Note: src.At(x, y).RGBA() returns values in the range [0, 65535]
    // so we need to convert them to [0, 255] by shifting right by 8 bits
    // (i.e., dividing by 256)
    // We then create a color.Gray with the computed Y' value
    for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
        for x := bounds.Min.X; x < bounds.Max.X; x++ {
            r, g, b, _ := src.At(x, y).RGBA()
            // convert from 16-bit to 8-bit
            R := float64(r >> 8)
            G := float64(g >> 8)
            B := float64(b >> 8)
            // luminance formula
            Y := 0.299*R + 0.587*G + 0.114*B
            dst.SetGray(x, y, color.Gray{Y: uint8(Y)})
        }
    }

    return dst
}
```

##### Original input image:
![original](assets/apple.jpg)
##### Gray Scale Image:
![gray_apple](output/gray_apple.png)

2) Add white gaussian noise to the grayscale image. Show the effects of the noise when
the standard deviation is 1, 10 and 30 and 50.

**Approach:**
For each gray pixel $p∈[0,255]$, sample n:
```go
n := rng.NormFloat64()*sigma 
```
And set the pixel of the new image:
```go
p' = clamp(p + n, 0, 255).
```

```go
func AddGaussianNoise(src *image.Gray, sigma float64, rng *rand.Rand) *image.Gray {
    b := src.Bounds()
    dst := image.NewGray(b)

    for y := b.Min.Y; y < b.Max.Y; y++ {
        // Row offsets for fast access
        srcOff := (y - b.Min.Y) * src.Stride
        dstOff := (y - b.Min.Y) * dst.Stride

        for x := b.Min.X; x < b.Max.X; x++ {
            p := float64(src.Pix[i])

            // NormFloat64 returns a normally distributed float64 in
            // the range -math.MaxFloat64 through +math.MaxFloat64 inclusive,
            // with standard normal distribution (mean = 0, stddev = 1).
            // To produce a different normal distribution, callers can
            // adjust the output using: sample = NormFloat64() * desiredStdDev + desiredMean
            // Sample N(0, sigma)
            n := rng.NormFloat64() * sigma

            v := p + n
            if v < 0 {
                v = 0
            } else if v > 255 {
                v = 255
            }
            dst.Pix[dstOff+(x-b.Min.X)] = uint8(v)
        }
    }
    return dst
}
```


##### Gaussian Noise with σ=1
![](output/gauss_sigma_1.png "Gaussian Noise with σ=1")

##### Gaussian Noise with σ=10
![](output/gauss_sigma_10.png "Gaussian Noise with σ=10")

##### Gaussian Noise with σ=30
![](output/gauss_sigma_30.png "Gaussian Noise with σ=30")

##### Gaussian Noise with σ=50
![](output/gauss_sigma_50.png "Gaussian Noise with σ=50")

**Approach:**
Randomly replace a fraction of pixels with 0 (pepper) or 255 (salt), approximately half each.
We iterate over every pixel in the gray scale image. At each pixel, we roll a number $r∈[0,255]$ and check if $r$ is less than the density parameter. 

If so, we check if $r < density/2$ and set the current pixel to 0 (pepper) if it is.
If $r$ is less than density but greater than $density/r$, set the current pixel to 255 (salt).

Otherwise, just write the pixel value from the original source image onto our new image.

```go
func AddSaltAndPepper(src *image.Gray, density float64, rng *rand.Rand) *image.Gray {
    rect := src.Bounds()
    dst := image.NewGray(rect)
    for y := 0; y < rect.Dy(); y++ {
        for x := 0; x < rect.Dx(); x++ {
            i := y*src.Stride + x
            val := src.Pix[i]

            r := rng.Float64() // returns a number [0, 1)
            if r < density/2 { // ~half peper
                val = 0
            } else if r < density { // ~half salt
                val = 255
            }
            dst.Pix[y*dst.Stride+x] = val
        }
    }
    return dst
}

```
##### Salt and Pepper 10%
![](output/saltpepper_10.png "Salt and Pepper 10%")

##### Salt and Pepper 30%
![](output/saltpepper_30.png "Salt and Pepper 30%")

4) Implement a Box Filter, Median Filter and Gaussian Filter to remove the white gaussian
noise with standard deviation 50 and the Salt and Pepper noise at 30%. Let the kernel
size be 3 x 3. Explain your choice of other parameters and any non-trivial steps in your
implementation (e.g., how did you handle the borders in the image).

**Approach:**
To implement the Box Filter, Median Filter and Gaussian Filter, I defined a `Filter` interface.
The `Filter` interface defines two methods `Name()` and `Apply()`.
- Name() – returns a short string describing the filter and its parameters (e.g., "gauss-5x5-sigma1.00").
         - This is useful for labeling output files automatically.

- Apply(src *image.Gray) – performs the actual filtering on a grayscale image 
                         - returns a new filtered image of the same size.

Types `BoxFilter`, `MedianFilter` and `GaussianFilter` implement the `Filter` interface.
```go
type Filter interface {
    Name() string
    Apply(src *image.Gray) *image.Gray
}

type BoxFilter struct {
    KernelSize int
}

type MedianFilter struct {
    KernelSize int
}

type GaussianFilter struct {
    KernelSize int
    Sigma      float64
}
```

I implemented a helper function `getKernelPixels` to extract the pixel values in the kernel around a given pixel.
It takes the source image, the x and y position of the center pixel, and the kernel size as input.
It returns a slice of pixel values in the kernel. This will be used by our filters to get the pixel values in the kernel.

When iterating over the kernel, we handle the borders by clamping the indices to the image bounds. We check if the kernel goes out of bounds, and if it does, we use the nearest edge pixel.
At each position in the kernel, we add the pixel value to `values`.
Once the kernel is processed, we return the `values` slice to be used by the filters.
```go 
func getKernelPixels(src *image.Gray, x_pos, y_pos, k int) []uint8 {
    w, h := src.Bounds().Dx(), src.Bounds().Dy() // width and height of the image in pixels
    r := k / 2 // radius of the kernel
    values := make([]uint8, 0, k*k)
    // Iterate over filter kernel
    // if the kernel goes out of bounds, we use the nearest edge pixel
    // This is done by clamping the indices to the image bounds
    // For even kernel sizes, we use floor(k/2) as the radius which means
    // the kernel will be slightly biased towards the top-left corner.
    // For example, for a 4x4 kernel, the radius is 2, so the kernel will cover
    // the pixels from (x-2, y-2) to (x+1, y+1)
    for j := range k {
        dy := j - r
        yy := y_pos + dy
        if yy < 0 {
            yy = 0
        } else if yy >= h {
            yy = h - 1
        }
        row := yy * src.Stride
        for i := range k {
            dx := i - r
            xx := x_pos + dx
            if xx < 0 {
                xx = 0
            } else if xx >= w {
                xx = w - 1
            }
            values = append(values, src.Pix[row+xx])
        }
    }
    return values
}
```

Below is the implementation of the Box Filter. The Box Filter computes the average of the pixels in the kernel
and assigns it to the center pixel. 

First, we create a new destination image of the same dimensions as the source image.
We take the width and height of the image which will be used to iterate over each pixel of the source image.
These steps will be the same for all three filters.
```go
func (f BoxFilter) Apply(src *image.Gray) *image.Gray {
    rect := src.Bounds()
    dst := image.NewGray(rect)
    w := rect.Dx()
    h := rect.Dy()
```

Next, we iterate over every pixel in the source image.
At each pixel, we set a `sum` variable to accumulate the pixel values in the kernel.
```go
    sum := 0
    for y := range h {
        for x := range w {
            sum = 0
```

Inside the pixel loop, we call `getKernelPixels` and store the resulting pixel slice in `kernel_pixels`.
We then sum the pixel values in the kernel and compute the average by dividing the `sum` by the total number of pixels in the kernel.
We set the pixel in the destination image to the computed average and repeat for all pixels.
And finally, we return the filtered image.
```go 
    for y := range h {
        for x := range w {
            kernal_pixels := getKernelPixels(src, x, y, f.KernelSize)
            sum := 0
            for _, v := range kernal_pixels {
                sum += int(v)
            }
            // Compute average and set pixel
            avg := sum / len(kernal_pixels)
            dst.Pix[y*dst.Stride+x] = uint8(avg)
        }
    }
    return dst
}
```

Since the `MedianFilter` implementation is very similar to the `BoxFilter`, I will only highlight the differences.
The main difference is that instead of computing the average of the pixels in the kernel, we compute the median.
For this, we simply take the pixel values in the kernel as a slice, sort it, and find the median.
Set the pixel in the destination image to the computed median and repeat for all pixels.
```go 
            kernel_pixels := getKernelPixels(src, x, y, f.KernelSize)
            // Compute median and set pixel
            slices.Sort(kernel_pixels)
            median := kernel_pixels[len(kernel_pixels)/2]
            dst.Pix[y*dst.Stride+x] = median
        }
    }
    return dst
}

```

The `GaussianFilter` implementation is also similar to the `BoxFilter` but slightly more complex.
First, we need to build the Gaussian kernel based on the specified `KernelSize` and `Sigma`.
The Gaussian kernel is a 1D array of weights that will be used to compute the weighted average of the pixels in the kernel.

Below is the `buildGaussianKernel` function that constructs the Gaussian kernel.
We first check if `sigma` is less than or equal to 0, and if so, we set it to `KernelSize/6.0`.
We then create a slice `kernel` to hold the kernel values and a `sum` variable to accumulate the sum of the kernel values for normalization.
If we do not normalize, the resulting pixel values may be too bright or too dark.
To normalize the kernel, we divide each value in the kernel by the sum of all the values.
The kernel is built using two nested loops that iterate over the kernel size and compute the Gaussian value for each position in the kernel using the Gaussian formula.
The resulting output is a 1D array of size `KernelSize * KernelSize` that stores the weight each pixel in the kernel contributes to the final value of the center pixel.
```go 
func buildGaussianKernel(k int, sigma float64) []float64 {
    if sigma <= 0 {
        sigma = float64(k) / 6.0
    }
    r := k / 2
    kernel := make([]float64, k*k)
    sum := 0.0
    idx := 0
    // build the Gaussian kernel!
    // G(x, y) = (1 / (2 * π * σ^2)) * exp(-(x^2 + y^2) / (2 * σ^2))
    //\(G(x,y)\) represents the value of the Gaussian kernel at coordinates \((x,y)\) relative to the center of the kernel.
    for y := 0; y < k; y++ {
        dy := y - r
        for x := 0; x < k; x++ {
            dx := x - r
            v := math.Exp(-(float64(dx*dx + dy*dy)) / (2.0 * sigma * sigma)) // unnormalized
            kernel[idx] = v
            sum += v
            idx++
        }
    }
    // Normalize the kernel so that the sum is 1
    inv := 1.0 / sum
    for i := range kernel {
        kernel[i] *= inv
    }
    return kernel
}
```

Now we can implement the `Apply` method for the `GaussianFilter`.
It starts similarly to the other filters by creating a new destination image and getting the width and height of the image.
After setting up the destination image, we build the Gaussian kernel using the `buildGaussianKernel` function.
```go 
func (f GaussianFilter) Apply(src *image.Gray) *image.Gray {
    rect := src.Bounds()
    dst := image.NewGray(rect)
    w := rect.Dx()
    h := rect.Dy()
    k := f.KernelSize

    wieghts := buildGaussianKernel(k, f.Sigma)
```

We iterate over each pixel in the source image and apply the Gaussian filter.
Since the kernel `weights` is already normalized, we can directly use it to compute the weighted sum of the pixels within the kernel's range. 
Since `kernel_pixels` and `weights` are both 1D arrays of the same length, we can simply multiply each pixel value by its corresponding weight and accumulate the result in `kernel_sum`.
The `kernel_sum` is then assigned to the pixel in the destination image.
```go 
    for y := 0; y < h; y++ {
        for x := 0; x < w; x++ {
            kernal_pixels := getKernelPixels(src, x, y, f.KernelSize)
            kernel_sum := 0.0
            for i, v := range kernal_pixels {
                kernel_sum += float64(v) * wieghts[i]
            }
            // debugging
            // if kernel_sum < 0 {
            //  fmt.Println("negative value in gaussian filter:", kernel_sum)
            //  kernel_sum = 0
            // } else if kernel_sum > 255 {
            //  fmt.Println("overflow value in gaussian filter:", kernel_sum)
            //  kernel_sum = 255
            // }
            dst.Pix[y*dst.Stride+x] = uint8(math.Round(kernel_sum))
        }
    }
    return dst
}
```

##### Salt-Pepper-Noise-30 BoxFilter-3x3
![Salt-Pepper-Noise-30 BoxFilter-3x3](output/saltpepper_30-box-3x3.png)

##### Salt-Pepper-Noise-30 MedianFilter-3x3
![Salt-Pepper-Noise-30 MedianFilter-3x3](output/saltpepper_30-median-3x3.png)

##### Salt-Pepper-Noise-30 GaussianFilter-3x3-sigma-1
![Salt-Pepper-Noise-30 GaussianFilter-3x3-sigma-1](output/saltpepper_30-gauss-3x3-sigma1.00.png)

##### White-Gauss-50 BoxFilter-3x3
![White-Gauss-50 BoxFilter-3x3](output/gauss_sigma_50-box-3x3.png)

##### White-Gauss-50 MedianFilter-3x3
![White-Gauss-50 MedianFilter-3x3](output/gauss_sigma_50-median-3x3.png)

##### White-Gauss-50 GaussianFilter-3x3-sigma-1
![White-Gauss-50 GaussianFilter-3x3-sigma-1](output/gauss_sigma_50-gauss-3x3-sigma1.00.png)

5) Vary the kernel size to 5 x 5 and 10 x 10 and show its effect on the output image.
#### 5x5 Kernel:
##### Salt-Pepper-Noise-30 BoxFilter-5x5
![Salt-Pepper-Noise-30 BoxFilter-5x5](output/saltpepper_30-box-5x5.png)

##### Salt-Pepper-Noise-30 MedianFilter-5x5
![Salt-Pepper-Noise-30 MedianFilter-5x5](output/saltpepper_30-median-5x5.png)

##### Salt-Pepper-Noise-30 GaussianFilter-5x5-sigma-1
![Salt-Pepper-Noise-30 GaussianFilter-5x5-sigma-1](output/saltpepper_30-gauss-5x5-sigma1.00.png)

##### White-Gauss-50 BoxFilter-5x5
![White-Gauss-50 BoxFilter-5x5](output/gauss_sigma_50-box-5x5.png)

##### White-Gauss-50 MedianFilter-5x5
![White-Gauss-50 MedianFilter-5x5](output/gauss_sigma_50-median-5x5.png)

##### White-Gauss-50 GaussianFilter-5x5-sigma-1
![White-Gauss-50 GaussianFilter-5x5-sigma-1](output/gauss_sigma_50-gauss-5x5-sigma1.00.png)

##### 10x10 Kernel:
##### Salt-Pepper-Noise-30 BoxFilter-10x10
![Salt-Pepper-Noise-30 BoxFilter-10x10](output/saltpepper_30-box-10x10.png)

##### Salt-Pepper-Noise-30 MedianFilter-10x10
![Salt-Pepper-Noise-30 MedianFilter-10x10](output/saltpepper_30-median-10x10.png)

##### Salt-Pepper-Noise-30 GaussianFilter-10x10-sigma-1
![Salt-Pepper-Noise-30 GaussianFilter-10x10-sigma-1](output/saltpepper_30-gauss-10x10-sigma1.00.png)

##### White-Gauss-50 BoxFilter-10x10
![White-Gauss-50 BoxFilter-10x10](output/gauss_sigma_50-box-10x10.png)

##### White-Gauss-50 MedianFilter-10x10
![White-Gauss-50 MedianFilter-10x10](output/gauss_sigma_50-median-10x10.png)

##### White-Gauss-50 GaussianFilter-10x10-sigma-1
![White-Gauss-50 GaussianFilter-10x10-sigma-1](output/gauss_sigma_50-gauss-10x10-sigma1.00.png)
