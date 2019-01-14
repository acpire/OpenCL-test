#pragma once
#include "clDevice.h"
class MakeNoise
{
	cl_float4* inverse_matrix(cl_float4* matrix, size_t width, size_t height);
	cl_float4* make_kernel_normal_distribution(size_t width, size_t height);
	void call_opencl_function(size_t index_kernel, clDevice* device, cl_uint* indices_images, cl_uint* indices_arguments, size_t number_images, size_t number_arguments);
	cl_float4* kernel;
public:
	cl_float4* getKernel();
	MakeNoise(clDevice* device, cl_uchar4* image, size_t width, size_t height, size_t width_filter, size_t height_filter);
	~MakeNoise();
};

