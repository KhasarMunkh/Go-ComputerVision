package main

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg" // ✅ registers JPEG decoder (optional)
	"image/png"    // ✅ gives access to png.Encode
	"log"
	"math/rand"
	"os"
)

var rng = rand.New(rand.NewSource(42))

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

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

func saveImage(img image.Image, path string) error {
	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer out.Close()
	return png.Encode(out, img)
}

// AddGaussianNoise returns a new *image.Gray where each pixel p is replaced by clamp(p + N(0, sigma)).
func AddGaussianNoise(src *image.Gray, sigma float64, rng *rand.Rand) *image.Gray {
	b := src.Bounds()
	dst := image.NewGray(b)

	for y := b.Min.Y; y < b.Max.Y; y++ {
		// Row offsets for fast access
		srcOff := (y - b.Min.Y) * src.Stride
		dstOff := (y - b.Min.Y) * dst.Stride

		for x := b.Min.X; x < b.Max.X; x++ {
			i := srcOff + (x - b.Min.X)
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

func question1(path string, out string) {
	original_img, err := loadImage(path)
	if err != nil {
		log.Fatal(err)
	}
	gray_image := toGray(original_img)
	err = saveImage(gray_image, out)
	if err != nil {
		log.Fatal(err)
	}
}

func question2(gray_image *image.Gray) {
	rng := rand.New(rand.NewSource(42))

	sigmas := []float64{1, 10, 30, 50}
	for _, s := range sigmas {
		noisy := AddGaussianNoise(gray_image, s, rng)
		outPath := fmt.Sprintf("output/gauss_sigma_%d.png", int(s))
		if err := saveImage(noisy, outPath); err != nil {
			log.Fatal(err)
		}
		log.Println("Saved:", outPath)
	}
}

// Add Salt and Pepper noise to 10% and 30% of the pixels in the grayscale image in 1.
// Salt and Pepper noise is added to an image by adding random bright and random dark
// pixels all over the image
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

func question3(gray_image *image.Gray) {
	densities := []float64{0.10, 0.30}
	for _, d := range densities {
		sp := AddSaltAndPepper(gray_image, d, rng)
		outPath := fmt.Sprintf("output/saltpepper_%.0f.png", d*100)
		if err := saveImage(sp, outPath); err != nil {
			log.Fatal(err)
		}
		log.Println("Saved:", outPath)
	}
}
func question4(gray_image *image.Gray, out_prefix string) {
	bf := BoxFilter{KernelSize: 3}
	mf := MedianFilter{KernelSize: 3}
	gf := GaussianFilter{KernelSize: 3, Sigma: 1}
	for _, f := range []Filter{bf, mf, gf} {
		out := f.Apply(gray_image)
		outPath := fmt.Sprintf("%s-%s.png", out_prefix, f.Name())
		if err := saveImage(out, outPath); err != nil {
			log.Fatal(err)
		}
		log.Println("Saved:", outPath)
	}
}

func question5(gray_image *image.Gray, out_prefix string) {
	bf := BoxFilter{KernelSize: 5}
	mf := MedianFilter{KernelSize: 5}
	gf := GaussianFilter{KernelSize: 5, Sigma: 1}
	for _, f := range []Filter{bf, mf, gf} {
		out := f.Apply(gray_image)
		outPath := fmt.Sprintf("%s-%s.png", out_prefix, f.Name())
		if err := saveImage(out, outPath); err != nil {
			log.Fatal(err)
		}
		log.Println("Saved:", outPath)
	}
	bf = BoxFilter{KernelSize: 10}
	mf = MedianFilter{KernelSize: 10}
	gf = GaussianFilter{KernelSize: 10, Sigma: 1}
	for _, f := range []Filter{bf, mf, gf} {
		out := f.Apply(gray_image)
		outPath := fmt.Sprintf("%s-%s.png", out_prefix, f.Name())
		if err := saveImage(out, outPath); err != nil {
			log.Fatal(err)
		}
		log.Println("Saved:", outPath)
	}
}

func main() {
	gray_img_gauss_sigma_50, err := loadImage("output/gauss_sigma_50.png")
	if err != nil {
		log.Fatal(err)
	}
	gray_img_saltpepper_30, err := loadImage("output/gauss_sigma_50.png")
	if err != nil {
		log.Fatal(err)
	}
	// question1("assets/apple.jpg", "output/gray_apple.png")
	// question2(gray_img.(*image.Gray))
	// question3(gray_img.(*image.Gray))
	// question4(gray_img.(*image.Gray), "output/gauss_sigma_50")
	// question4(gray_img.(*image.Gray), "output/saltpepper_30")
	question5(gray_img_gauss_sigma_50.(*image.Gray), "output/gauss_sigma_50")
	question5(gray_img_saltpepper_30.(*image.Gray), "output/saltpepper_30")
}
