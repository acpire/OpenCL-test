#include "Lab_2.h"

void Lab_2::convolution_rgba_host(size_t index_kernel, clDevice* device, cl_uchar4* image, size_t width, size_t height, cl_float4* kernel, size_t width_kernel, size_t height_kernel) {
	void* data[] = { (void*)image, (void*)kernel, NULL };
	cl_uchar type_arguments[] = { sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint) };
	size_t length_data[] = { height * width * sizeof(cl_uchar4), width_kernel * height_kernel * sizeof(cl_float4), height * width * sizeof(cl_uchar4) };
	cl_uint indices_memory[3];
	cl_uint index_kernel_buffer[] = { 0, 3, 6 };
	cl_uint index_kernel_arguments[] = { 1, 2, 4, 5 };
	size_t work_size[] = { width, height, 1 };
	cl_uint arguments[] = { width,  height , width_kernel, height_kernel };
	size_t indices = device->mallocBufferMemory((const void**)data, length_data, 3, sizeof(char));
	indices_memory[0] = indices;
	indices_memory[1] = indices + 1;
	indices_memory[2] = indices + 2;
	device->setArguments(index_kernel, indices_memory, 3, index_kernel_buffer, arguments, type_arguments, 4, index_kernel_arguments);
	device->startCalculate(index_kernel, work_size);
	number_buffers += 3;
	length_last_buffer = 3;
	const char* buffers[3] = { "image_buffer" + 0,"kernel_buffer" + 0, "convolution_image" + 0 };
	const size_t length_buffers[] = { 14, 15, 19 };
	if (types_buffers_OpenCL)
		types_buffers_OpenCL = (cl_uchar**)realloc(types_buffers_OpenCL, number_buffers * sizeof(cl_uchar*));
	else
		types_buffers_OpenCL = (cl_uchar**)malloc(number_buffers * sizeof(cl_uchar*));

	length_buffer_OpenCL = (size_t*)realloc(length_buffer_OpenCL, number_buffers * sizeof(size_t));
	indices_buffer_OpenCL = (size_t*)realloc(indices_buffer_OpenCL, number_buffers * sizeof(size_t));
	last_buffers_modify = (cl_uint*)realloc(last_buffers_modify, length_last_buffer * sizeof(cl_uint));
	for (size_t i = number_buffers - 3, j = 0; i < number_buffers; i++, j++) {
		types_buffers_OpenCL[i] = (cl_uchar*)malloc(length_buffers[i] * sizeof(char));

		memcpy(types_buffers_OpenCL[i], buffers[i], length_buffers[i] * sizeof(char));
		indices_buffer_OpenCL[i] = indices + i;
		length_buffer_OpenCL[i] = length_buffers[i];
		last_buffers_modify[j] = number_buffers - length_last_buffer + i;
	}
}
void Lab_2::convolution_rgba_device(size_t index_kernel, clDevice* device, size_t width, size_t height, size_t width_kernel, size_t height_kernel, cl_uint indices_buffers[3]) {
	cl_uchar type_arguments[] = { sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint) };
	cl_uint index_kernel_buffer[] = { 0, 3, 6 };
	cl_uint index_kernel_arguments[] = { 1, 2, 4, 5 };
	size_t work_size[] = { width, height, 1 };
	cl_uint arguments[] = { width,  height , width_kernel, height_kernel };
	device->setArguments(index_kernel, indices_buffers, 3, index_kernel_buffer, arguments, type_arguments, 4, index_kernel_arguments);
	device->startCalculate(index_kernel, work_size);
	length_last_buffer = 3;
	last_buffers_modify = (cl_uint*)realloc(last_buffers_modify, length_last_buffer * sizeof(cl_uint));
	for (size_t i = 0; i < 3; i++) {
		last_buffers_modify[i] = indices_buffers[i];
	}
}

void Lab_2::make_noise_rgba_host(size_t index_kernel, clDevice* device, cl_uchar4* image, size_t width, size_t height, float mathematical_expectation, float standard_deviation) {
	void* data[] = { (void*)image, (void*)NULL };
	cl_uchar type_arguments[] = { sizeof(cl_uint),  sizeof(cl_uint),  sizeof(float),  sizeof(float) };
	size_t length_data[] = { height * width * sizeof(cl_uchar4), height * width * sizeof(cl_uchar4) };
	cl_uint indices_memory[2];
	cl_uint index_kernel_buffer[] = { 0, 1 };
	cl_uint index_kernel_arguments[] = { 2, 3, 4, 5 };
	size_t work_size[] = { width, height, 1 };
	cl_uint arguments[] = { width,  height, *((cl_uint*)&mathematical_expectation), *((cl_uint*)&standard_deviation) };
	size_t indices = device->mallocBufferMemory((const void**)data, length_data, 2, sizeof(char));
	indices_memory[0] = indices;
	indices_memory[1] = indices + 1;
	device->setArguments(index_kernel, indices_memory, 2, index_kernel_buffer, arguments, type_arguments, 4, index_kernel_arguments);
	device->startCalculate(index_kernel, work_size);
	number_buffers += 2;
	length_last_buffer = 2;
	const char* buffers[2] = { "image_buffer" + 0, "noise_image_buffer" + 0 };
	const size_t length_buffers[] = { 20 };
	types_buffers_OpenCL = (cl_uchar**)realloc(types_buffers_OpenCL, number_buffers * sizeof(cl_uchar*));
	length_buffer_OpenCL = (size_t*)realloc(length_buffer_OpenCL, number_buffers * sizeof(size_t));
	indices_buffer_OpenCL = (size_t*)realloc(indices_buffer_OpenCL, number_buffers * sizeof(size_t));
	last_buffers_modify = (cl_uint*)realloc(last_buffers_modify, length_last_buffer * sizeof(cl_uint));
	for (size_t i = number_buffers - 2, j = 0; i < number_buffers; i++, j++) {
		types_buffers_OpenCL[i] = (cl_uchar*)malloc(length_buffers[i] * sizeof(char));
		memcpy(types_buffers_OpenCL[i], buffers[i], length_buffers[i] * sizeof(char));
		indices_buffer_OpenCL[i] = indices + i;
		length_buffer_OpenCL[i] = length_buffers[i];
		last_buffers_modify[j] = number_buffers - length_last_buffer + i;
	}
}

void Lab_2::make_noise_rgba_device(size_t index_kernel, clDevice* device, cl_uchar4* image, size_t width, size_t height, float mathematical_expectation, float standard_deviation, cl_uint indices_buffers[2]) {
	void* data[] = { (void*)image, (void*)NULL };
	cl_uchar type_arguments[] = { sizeof(cl_uint),  sizeof(cl_uint),  sizeof(float),  sizeof(float) };
	cl_uint index_kernel_buffer[] = { 0, 1 };
	cl_uint index_kernel_arguments[] = { 2, 3, 4, 5 };
	size_t work_size[] = { width, height, 1 };
	cl_uint arguments[] = { width,  height, *((cl_uint*)&mathematical_expectation), *((cl_uint*)&standard_deviation) };
	device->setArguments(index_kernel, indices_buffers, 2, index_kernel_buffer, arguments, type_arguments, 4, index_kernel_arguments);
	device->startCalculate(index_kernel, work_size);
	length_last_buffer = 2;
	last_buffers_modify = (cl_uint*)realloc(last_buffers_modify, length_last_buffer * sizeof(cl_uint));
	for (size_t i = 0; i < number_buffers; i++) {
		last_buffers_modify[i] = indices_buffers[i];
	}
}
cl_float4* Lab_2::make_kernel_normal_distribution(size_t width, size_t height) {
	cl_float4* kernel = (cl_float4*)_alloca(width * height * sizeof(cl_float4));
	float math_1 = 0;
	float math_2 = 0;
	float D_1 = 0;
	float D_2 = 0;
	float correlation = 0;
	for (size_t y = 0; y < height; y++)
		math_2 += y;
	for (size_t x = 0; x < width; x++)
		math_1 += x;
	math_1 /= width;
	math_2 /= height;
	for (size_t x = 0; x < width; x++)
		D_1 += pow(x - math_1, 2);
	for (size_t y = 0; y < height; y++)
		D_2 += pow(y - math_2, 2);
	D_1 /= width;
	D_2 /= height;
	for (size_t x = 0; x < width; x++)
		correlation += ((x - math_1) * (x - math_2));
	correlation /= (D_1 * D_2);
	float C = 1.0f / (2.0f * CL_M_PI * D_1 * D_2 * sqrtf(1.0f - correlation * correlation));
	cl_double sum = 0;
	for (size_t y = 0; y < height; y++) {
		for (size_t x = 0; x < width; x++) {
			float Q = (1.0f / (2.0f * 1.0f - correlation * correlation)) * ((powf(x - math_1, 2) / D_1) + (powf(y - math_2, 2) / D_2) + (2.0f * correlation * (x - math_1) * (y - math_2) / D_1 * D_2));
			cl_float data = C * expf(-Q);
			sum += data;
			kernel[y * width + x] = cl_float4{ data, data, data, data };
		}
	}
	sum = height * width / sum;
	for (size_t i = 0; i < height * width; i++) {
		kernel[i].x *= sum;
		kernel[i].y *= sum;
		kernel[i].z *= sum;
		kernel[i].w *= sum;
		kernel[i].x = 1;
		kernel[i].y = 1;
		kernel[i].z = 1;
		kernel[i].w = 1;
		printf("%f ", kernel[i].x);
		if (i % width == 0)
			printf("\n");
	}
	return kernel;
}
Lab_2::Lab_2(clDevice* device, cl_uchar4* image, size_t width, size_t height, size_t* indices_kernels, size_t number_kernels)
{
	length_buffer_OpenCL = 0;
	indices_buffer_OpenCL = 0;
	index_kernel = 0;
	number_buffers = 0;
	length_last_buffer = 0;
	length_kernel = 0;
	last_buffers_modify = 0;
	types_buffers_OpenCL = 0;
	cl_float4* kernel = make_kernel_normal_distribution(20, 20);

	/*const int part_height_kernel = 20 >> 1;
	const int part_width_kernel = 20 >> 1;
	const float operations = (20 * 20);
	for (int h = 0; h < height; h += 1) {
		for (int w = 0; w < width; w += 1) {
			float sumr = 0.0f;
			float sumg = 0.0f;
			float sumb = 0.0f;
			float suma = 0.0f;
			for (int i = 0; i < 20; i++) {
				for (int j = 0; j < 20; j++) {
					const int index_y = (h + i - part_height_kernel);
					const int index_x = (w + j - part_width_kernel);
					if (index_y >= 0 && index_y < height && index_x >= 0 && index_x < width) {
						sumr += (cl_float)image[index_y * width + index_x].x  * (cl_float)kernel[i * 20 + j].x;
						sumg += (cl_float)image[index_y * width + index_x].y  * (cl_float)kernel[i * 20 + j].y;
						sumb += (cl_float)image[index_y * width + index_x].z  * (cl_float)kernel[i * 20 + j].z;
						suma += (cl_float)image[index_y * width + index_x].w  * (cl_float)kernel[i * 20 + j].w;
					}
					else {
						sumr += 0;
						sumg += 0;
						sumb += 0;
						suma += 0;
					}
				}
			}
			sumr /= operations;
			sumg /= operations;
			sumb /= operations;
			suma /= operations;
			image[h * width + w].x = (sumr);
			image[h * width + w].y = (sumg);
			image[h * width + w].z = (sumb);
			image[h * width + w].w = (suma);
		}
	}*/

	cl_uchar type_arguments[] = { sizeof(cl_uint) };
	size_t length_data[] = { height * width * sizeof(cl_uchar4) };

	convolution_rgba_host(0, device, image, width, height, kernel, 20, 20);

	device->readData((void**)&image, &last_buffers_modify[length_last_buffer - 1], type_arguments, length_data, 1);
}

Lab_2::~Lab_2()
{
	for (size_t i = 0; i < number_buffers; i++) {
		free(types_buffers_OpenCL[i]);
	}
	if (last_buffers_modify)
		free(last_buffers_modify);
	if (index_kernel)
		free(index_kernel);
	if (types_buffers_OpenCL)
		free(types_buffers_OpenCL);
	if (indices_buffer_OpenCL) {
		free(indices_buffer_OpenCL);
	}
}