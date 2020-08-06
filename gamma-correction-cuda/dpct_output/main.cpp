//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <fstream>
#include "utils.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

using namespace std;


void gamma_correction(ImgPixel* pixel, sycl::nd_item<3> item_ct1) {
  int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);

        // Lambda to process image with gamma = 2
	const float v = (0.3f * pixel[i].r + 0.59f * pixel[i].g + 0.11f * pixel[i].b) / 255.0;

	std::uint8_t gamma_pixel = static_cast<std::uint8_t>(255 * v * v);
	if (gamma_pixel > 255) gamma_pixel = 255;
	pixel[i].set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        // Image size is width x height
	int width = 2560;
	int height = 1600;

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

	ImgPixel* pixel;
  pixel = (ImgPixel *)sycl::malloc_device(
      sizeof(ImgPixel) * image2.width() * image2.height(), q_ct1);
  q_ct1
      .memcpy(pixel, image2.data(),
              sizeof(ImgPixel) * image2.width() * image2.height())
      .wait();
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, width * height / BLOCK_SIZE) *
                              sycl::range<3>(1, 1, BLOCK_SIZE),
                          sycl::range<3>(1, 1, BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) { gamma_correction(pixel, item_ct1); });
  });
  q_ct1
      .memcpy(image2.data(), pixel,
              sizeof(ImgPixel) * image2.width() * image2.height())
      .wait();

        // check correctness
	if (check(image.begin(), image.end(), image2.begin())) {
		std::cout << "Pass";
	} else {
		std::cout << "Fail";
	}

  dpct::device_info props;
  dpct::dev_mgr::instance().get_device(0).get_device_info(props);

  std::cout << ". Run on " << props.get_name() << std::endl;

#ifdef DEBUG
	image.write("fractal_gamma_parallel.bmp");
#endif
  sycl::free(pixel, q_ct1);

        return 0;
}
