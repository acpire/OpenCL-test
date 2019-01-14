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
	void call_opencl_function(size_t index_kernel, clDevice* device, cl_uint* indices_images, cl_uint* indices_arguments, size_t number_images, size_t number_arguments);
	void readImage(clDevice* device, void* returnData, size_t width_image, size_t height_image, cl_uint index_buffer);
	size_t mallocMemoryImage(clDevice* device,const void* ptr, size_t width, size_t height, size_t type_data, size_t type_image);
	void convolution_rgba_device(size_t index_kernel, clDevice* device,  size_t width, size_t height, size_t width_kernel, size_t height_kernel, cl_uint indices_images[3]);
	void inverse_image_rgba_device(size_t index_kernel, clDevice* device, size_t width, size_t height, cl_uint indices_images[2]);
	cl_float4* inverse_matrix(cl_float4* matrix, size_t width, size_t height);
	void divide_fourier_transform_rgba_image_device(size_t index_kernel, clDevice* device, size_t width, size_t height, size_t width_kernel, size_t height_kernel, cl_uint indices_images[6]);
	void convert_float4_to_uchar4_image(size_t index_kernel, clDevice* device, size_t width, size_t height, cl_uint indices_images[2]);
	cl_float4* make_kernel_normal_distribution(size_t width, size_t height);
	void inverse_fourier_transform_rgba_image_device(size_t index_kernel, clDevice* device, size_t width, size_t height, cl_uint indices_images[2]);
	void make_noise_image_rgba_device(size_t index_kernel, clDevice* device, size_t width, size_t height, cl_uint indices_images[2]);
	void fourier_transform_rgba_image_device(size_t index_kernel, clDevice* device, size_t width, size_t height, cl_uint indices_images[2]);
	void convolution_rgba_host(size_t index_kernel, clDevice * device, cl_uchar4 * image, size_t width, size_t height, cl_float4 * kernel, size_t width_kernel, size_t height_kernel);
	void convolution_rgba_host_image(size_t index_kernel, clDevice * device, cl_uchar4 * image, size_t width, size_t height, cl_float4 * kernel, size_t width_kernel, size_t height_kernel);
	void make_noise_rgba_host(size_t index_kernel, clDevice * device, cl_uchar4 * image, size_t width, size_t height, float mathematical_expectation, float standard_deviation);
	void make_noise_rgba_device(size_t index_kernel, clDevice * device, size_t width, size_t height, float mathematical_expectation, float standard_deviation, cl_uint indices_buffers);
	void fourier_transform_rgba_device(size_t index_kernel, clDevice * device, size_t width, size_t height, cl_uint indices_buffers[2]);
	Lab_2(clDevice* device, cl_uchar4* image, size_t width, size_t height, cl_float4* kernel, size_t width_kernel, size_t height_kernel);
		~Lab_2();
};

