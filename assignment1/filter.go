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
	return fmt.Sprintf("box-%dx%d-sigma%.2f", f.KernelSize, f.KernelSize, f.Sigma)
}

func (f BoxFilter) Apply(src *image.Gray) *image.Gray {
	rect := src.Bounds()
	dst := image.NewGray(rect)
	w := rect.Dx()
	h := rect.Dy()
	r := f.KernelSize / 2 // radius = (kernel size - 1)/2

	for y := range h {
		for x := range w {
			sum := 0
			// Iterate over filter kernel
			// if the kernel goes out of bounds, we use the nearest edge pixel
			// This is done by clamping the indices to the image bounds
			for j := -r; j <= r; j++ {
				yy := y + j
				if yy < 0 {
					yy = 0
				} else if yy >= h {
					yy = h - 1
				}
				for i := -r; i <= r; i++ {
					xx := x + i
					if xx < 0 {
						xx = 0
					} else if xx >= w {
						xx = w - 1
					}
					sum += int(src.Pix[yy*src.Stride+xx])
				}
			}
			// Compute average and set pixel
			avg := sum / (f.KernelSize * f.KernelSize)
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
	r := f.KernelSize / 2 // radius = (kernel size - 1)/2
	window := make([]uint8, 0, f.KernelSize*f.KernelSize)
	for y := range h {
		for x := range w {
			window = window[:0]
			// Iterate over filter kernel
			// if the kernel goes out of bounds, we use the nearest edge pixel
			// This is done by clamping the indices to the image bounds
			for j := -r; j <= r; j++ {
				yy := y + j
				if yy < 0 {
					yy = 0
				} else if yy >= h {
					yy = h - 1
				}
				for i := -r; i <= r; i++ {
					xx := x + i
					if xx < 0 {
						xx = 0
					} else if xx >= w {
						xx = w - 1
					}
					window = append(window, src.Pix[yy*src.Stride+xx])
				}
			}
			// Compute median and set pixel
			slices.Sort(window)
			median := window[len(window)/2]
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
	r := f.KernelSize / 2 // radius = (kernel size - 1)/2

	// build the Gaussian kernel!
	// G(x, y) = (1 / (2 * π * σ^2)) * exp(-(x^2 + y^2) / (2 * σ^2))
	//\(G(x,y)\) represents the value of the Gaussian kernel at coordinates \((x,y)\) relative to the center of the kernel.
	gaussian_kernel := make([][]float64, f.KernelSize)
	for i := range gaussian_kernel {
		gaussian_kernel[i] = make([]float64, f.KernelSize)
	}
	sigma_squared := f.Sigma * f.Sigma
	kernal_sum := 0.0
	for j := -r; j <= r; j++ {
		for i := -r; i <= r; i++ {
			v := (1.0 / (2 * math.Pi * sigma_squared)) * (math.Exp(-(float64(i*i + j*j)) / (2 * sigma_squared)))
			gaussian_kernel[j+r][i+r] = v
			kernal_sum += v
		}
	}
	// Normalize the kernel using the sum of the kernel values
	for j := -r; j <= r; j++ {
		for i := -r; i <= r; i++ {
			gaussian_kernel[j+r][i+r] /= kernal_sum
		}
	}

	for y := range h {
		for x := range w {
			sum := 0
			// Iterate over filter kernel
			// if the kernel goes out of bounds, we use the nearest edge pixel
			// This is done by clamping the indices to the image bounds
			for j := -r; j <= r; j++ {
				yy := y + j
				if yy < 0 {
					yy = 0
				} else if yy >= h {
					yy = h - 1
				}
				for i := -r; i <= r; i++ {
					xx := x + i
					if xx < 0 {
						xx = 0
					} else if xx >= w {
						xx = w - 1
					}
					weight := gaussian_kernel[j+r][i+r]
					sum += int(float64(src.Pix[yy*src.Stride+xx]) * weight)
				}
			}
			if sum < 0 {
				fmt.Println("sum < 0")
				sum = 0
			} else if sum > 255 {
				fmt.Println("sum > 255")
				sum = 255
			}
			dst.Pix[y*dst.Stride+x] = uint8(sum)
		}
	}
	return dst
}

