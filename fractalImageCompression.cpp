#include "fractalImageCompression.h"


fractalImageCompression::fractalImageCompression(clDevice* device, cl_uchar4* image, size_t width, size_t height)
{
	const size_t size_domain = 8;
	const size_t size_rank = 4;
	const int idx = 300;
	const int idy = 0;
	const int stride_x = 1;
	const int stride_y = 1;
	const int number_rank_blocks_x = width / size_domain;
	const int number_rank_blocks_y = height / size_domain;
	float rank_block[size_domain][size_domain];
	const float scale[2][2] = { float(size_rank) / float(size_domain), 0, 0, float(size_rank) / float(size_domain) };

	const char transformation[16 * 16] = { 1,0,0,1,  0,1,-1,0, -1,0,0,-1, 0,-1,1,0, 0,-1,-1,0, -1,0,0,1, 0,1,1,0, 1,0,0,-1 };
	const float size_block = size_domain * size_domain;
	int domainBlock[2] = { 0, 0 };
	for (size_t i = idy; i < number_rank_blocks_y; i += stride_y) {
		for (size_t j = idy; j < number_rank_blocks_y; j += stride_x) {
			float mean_1 = (float)0;
			float mean_2 = (float)0;
			float min_disp = (float)CL_FLT_MAX;
			for (size_t index_y = 0; index_y < size_domain; index_y++)
				for (size_t index_x = 0; index_x < size_domain; index_x++) {
					rank_block[index_y][index_x] = image[j * size_domain + index_x + (i * size_domain + index_y) * width].x;
				}
			for (size_t h = 0; h < height - size_rank; h++) {
				for (size_t w = 0; w < width - size_rank; w++) {
					for (size_t index_number_transformations = 12; index_number_transformations < 8*4; index_number_transformations+=4) {

						float disp = (float)0;
						for (size_t b_y = 0; b_y < size_domain; b_y++) {
							for (size_t b_x = 0; b_x < size_domain; b_x++) {
								int x_index = (w + b_x);
								int y_index = (h + b_y);
								int index_image_x = x_index * scale[1][1] * transformation[index_number_transformations] + y_index * scale[0][0] * transformation[index_number_transformations + 1];
								int index_image_y = y_index * scale[0][0] * transformation[index_number_transformations + 3] + x_index * scale[1][1] *  transformation[index_number_transformations + 2];
								index_image_y = index_image_y < 0 ? size_rank + index_image_y : index_image_y;
								index_image_x = index_image_x < 0 ? size_rank + index_image_x : index_image_x;
								const float tmp = image[index_image_x + index_image_y * width].x;
								mean_1 += rank_block[b_y][b_x] - tmp;
								mean_2 += tmp;
								printf("%u ", index_image_x + index_image_y * width);
								
							}
							printf("\n");
						}
						mean_1 /= size_block;
						mean_2 /= size_block;
						for (size_t b_y = 0; b_y < size_domain; b_y++) {
							for (size_t b_x = 0; b_x < size_domain; b_x++) {
								int index_image = (w + b_x) * scale[1][1];
								index_image += (h + b_y) * scale[0][0] * width;
								const float tmp = image[index_image].x;
								disp += powf(rank_block[b_y][b_x] - tmp - mean_1, 2);
							}
						}
						disp /= size_block;
						if (disp < min_disp) {
							mean_1 = 0;
							domainBlock[0] = w;
							domainBlock[1] = h;
							for (size_t b_y = 0; b_y < size_domain; b_y++) {
								for (size_t b_x = 0; b_x < size_domain; b_x++) {
									int index_image = (w + b_x) * scale[1][1];
									index_image += (h + b_y) * scale[0][0] * width;
									const float tmp = image[index_image].x;
									mean_1 += rank_block[b_y][b_x];
									mean_2 += tmp;

								}
							}
						}
						mean_1 = 0;
					}
				}
			}
		}
	}

}


fractalImageCompression::~fractalImageCompression()
{
}
