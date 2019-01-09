#pragma once
#include "clDevice.h"
class Lab_2
{
	size_t* length_buffer_OpenCL;
	size_t* indices_buffer_OpenCL;
	size_t* index_kernel;
	size_t number_buffers;
	size_t length_last_buffer;
	size_t length_kernel;
	cl_uint* last_buffers_modify;
	cl_uchar** types_buffers_OpenCL;
public:
	cl_float4* make_kernel_normal_distribution(size_t width, size_t height);
	void convolution_rgba_host(size_t index_kernel, clDevice * device, cl_uchar4 * image, size_t width, size_t height, cl_float4 * kernel, size_t width_kernel, size_t height_kernel);
	void convolution_rgba_device(size_t index_kernel, clDevice * device, size_t width, size_t height, size_t width_kernel, size_t height_kernel, cl_uint indices_buffers[3]);
	void make_noise_rgba_host(size_t index_kernel, clDevice * device, cl_uchar4 * image, size_t width, size_t height, float mathematical_expectation, float standard_deviation);
	void make_noise_rgba_device(size_t index_kernel, clDevice * device, cl_uchar4 * image, size_t width, size_t height, float mathematical_expectation, float standard_deviation, cl_uint indices_buffers[2]);
	Lab_2(clDevice* device, cl_uchar4 * image, size_t width, size_t height, size_t * indices_kernels, size_t number_kernels);
	~Lab_2();
};

