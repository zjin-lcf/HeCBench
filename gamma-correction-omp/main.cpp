//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iomanip>
#include <iostream>
#include "utils.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

int main() {
	// Image size is width x height
	int width = 7680; //2560;
	int height = 4320; //1600;

	Img<ImgFormat::BMP> image{width, height};
	ImgFractal fractal{width, height};

	// Lambda to process image with gamma = 2
	auto gamma_f = [](ImgPixel& pixel) {
		float v = (0.3f * pixel.r + 0.59f * pixel.g + 0.11f * pixel.b) / 255.0;
		std::uint8_t gamma_pixel = static_cast<std::uint8_t>(255 * v * v);
		if (gamma_pixel > 255) gamma_pixel = 255;
		pixel.set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
	};

	// fill image with created fractal
	int index = 0;
	image.fill([&index, width, &fractal](ImgPixel& pixel) {
          int x = index % width;
          int y = index / width;
          
          auto fractal_pixel = fractal(x, y);
          if (fractal_pixel < 0) fractal_pixel = 0;
          if (fractal_pixel > 255) fractal_pixel = 255;
          pixel.set(fractal_pixel, fractal_pixel, fractal_pixel, fractal_pixel);
          
          ++index;
        });

	Img<ImgFormat::BMP> image2 = image;
#ifdef DEBUG
	image.write("fractal_original.bmp");
#endif

	// call standard serial function for correctness check
	image.fill(gamma_f);
#ifdef DEBUG
	image.write("fractal_gamma_serial.bmp");
#endif
	auto pixel = image2.data();
        int image_size = image2.width() * image2.height(); 

        #pragma omp target map(pixel[0:image_size])
	{
          #pragma omp teams distribute parallel for thread_limit(BLOCK_SIZE)
          for (int i = 0; i < image_size; i++) {
              // Lambda to process image with gamma = 2
              const float v = (0.3f * pixel[i].r + 0.59f * pixel[i].g + 0.11f * pixel[i].b) / 255.0;
              std::uint8_t gamma_pixel = static_cast<std::uint8_t>(255 * v * v);
              if (gamma_pixel > 255) gamma_pixel = 255;
              pixel[i].set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
	  }
	}

	// check correctness
	if (check(image.begin(), image.end(), image2.begin())) {
		std::cout << "Pass\n";
	} else {
		std::cout << "Fail\n";
	}

#ifdef DEBUG
	image.write("fractal_gamma_parallel.bmp");
#endif
	return 0;
}
