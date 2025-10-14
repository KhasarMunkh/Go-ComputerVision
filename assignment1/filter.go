package main

import (
	"fmt"
	"image"
	"math"
	"slices"
)

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

func (f BoxFilter) Name() string {
	return fmt.Sprintf("box-%dx%d", f.KernelSize, f.KernelSize)
}

func (f MedianFilter) Name() string {
	return fmt.Sprintf("median-%dx%d", f.KernelSize, f.KernelSize)
}

func (f GaussianFilter) Name() string {
	return fmt.Sprintf("gauss-%dx%d-sigma%.2f", f.KernelSize, f.KernelSize, f.Sigma)
}

func (f BoxFilter) Apply(src *image.Gray) *image.Gray {
	rect := src.Bounds()
	dst := image.NewGray(rect)
	w := rect.Dx()
	h := rect.Dy()

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

func (f MedianFilter) Apply(src *image.Gray) *image.Gray {
	rect := src.Bounds()
	dst := image.NewGray(rect)
	w := rect.Dx()
	h := rect.Dy()
	for y := range h {
		for x := range w {
			kernel_pixels := getKernelPixels(src, x, y, f.KernelSize)
			// Compute median and set pixel
			slices.Sort(kernel_pixels)
			median := kernel_pixels[len(kernel_pixels)/2]
			dst.Pix[y*dst.Stride+x] = median
		}
	}
	return dst
}

func (f GaussianFilter) Apply(src *image.Gray) *image.Gray {
	rect := src.Bounds()
	dst := image.NewGray(rect)
	w := rect.Dx()
	h := rect.Dy()
	k := f.KernelSize

	wieghts := buildGaussianKernel(k, f.Sigma)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			kernal_pixels := getKernelPixels(src, x, y, f.KernelSize)
			kernel_sum := 0.0
			for i, v := range kernal_pixels {
				kernel_sum += float64(v) * wieghts[i]
			}
			// debugging
			// if kernel_sum < 0 {
			//	fmt.Println("negative value in gaussian filter:", kernel_sum)
			//	kernel_sum = 0
			// } else if kernel_sum > 255 {
			//	fmt.Println("overflow value in gaussian filter:", kernel_sum)
			//	kernel_sum = 255
			// }
			dst.Pix[y*dst.Stride+x] = uint8(math.Round(kernel_sum))
		}
	}
	return dst
}

// Helper function to get pixels in a kernel
func getKernelPixels(src *image.Gray, x_pos, y_pos, k int) []uint8 {
	w, h := src.Bounds().Dx(), src.Bounds().Dy()
	r := k / 2
	values := make([]uint8, 0, k*k)
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

// builds a flattened Gaussian kernel of size k x k with standard deviation sigma.
// The kernel is normalized so that the sum of all its elements equals 1.
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
