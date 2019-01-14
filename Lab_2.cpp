#include "Lab_2.h"

void Lab_2::convolution_rgba_host_image(size_t index_kernel, clDevice* device, cl_uchar4* image, size_t width, size_t height, cl_float4* kernel, size_t width_kernel, size_t height_kernel) {
	void* data[] = { (void*)image, (void*)kernel, NULL };
	cl_uchar type_arguments[] = { sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint) };
	size_t length_height_data[] = { height ,  height_kernel , height };
	size_t length_width_data[] = { width , width_kernel , width };
	size_t length_row_pitch_data[] = { width * sizeof(cl_uchar4), width_kernel * sizeof(cl_float4), width * sizeof(cl_uchar4) };
	size_t type_image[] = { CL_RGBA, CL_RGBA, CL_RGBA };
	size_t type_data[] = { CL_UNSIGNED_INT8, CL_FLOAT, CL_UNSIGNED_INT8 };
	cl_uint indices_memory[3];
	cl_uint index_kernel_buffer[] = { 0, 3, 6 };
	cl_uint index_kernel_arguments[] = { 1, 2, 4, 5 };
	size_t work_size[] = { width, height, 1 };
	cl_uint arguments[] = { width,  height , width_kernel, height_kernel };
	size_t indices = device->mallocImageMemory((const void**)data, length_height_data, length_width_data, length_row_pitch_data, 3, type_image, type_data);
	indices_memory[0] = indices;
	indices_memory[1] = indices + 1;
	indices_memory[2] = indices + 2;
	device->setArguments(index_kernel, NULL, NULL, indices_memory, 3, index_kernel_buffer, arguments, type_arguments, 4, index_kernel_arguments);
	device->startCalculate(index_kernel, work_size);
	number_buffers += 3;
	length_last_buffer = 3;
	const char* buffers[3] = { "image_image2d" + 0,"kernel_image2d" + 0, "convolution_image2d" + 0 };
	const size_t length_buffers[] = { 14, 15, 19 };
	types_buffers_OpenCL = (cl_uchar**)realloc(types_buffers_OpenCL, number_buffers * sizeof(cl_uchar*));
	length_buffer_OpenCL = (size_t*)realloc(length_buffer_OpenCL, number_buffers * sizeof(size_t));
	indices_buffer_OpenCL = (size_t*)realloc(indices_buffer_OpenCL, number_buffers * sizeof(size_t));
	last_buffers_modify = (cl_uint*)realloc(last_buffers_modify, length_last_buffer * sizeof(cl_uint));
	for (size_t i = number_buffers - 3, j = 0; i < number_buffers; i++, j++) {
		types_buffers_OpenCL[i] = (cl_uchar*)malloc(length_buffers[j] * sizeof(char));

		memcpy(types_buffers_OpenCL[i], buffers[j], length_buffers[j] * sizeof(char));
		indices_buffer_OpenCL[i] = indices + j;
		length_buffer_OpenCL[i] = length_buffers[j];
		last_buffers_modify[j] = i;
	}
}
void Lab_2::call_opencl_function(size_t index_kernel, clDevice* device, cl_uint* indices_images, cl_uint* indices_arguments, size_t number_images, size_t number_arguments) {
	cl_uchar type_arguments[] = { sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint) };
	cl_uint* index_kernel_buffer = (cl_uint*)_alloca(number_images*sizeof(cl_uint));
	cl_uint* index_kernel_arguments = (cl_uint*)_alloca(number_arguments * sizeof(cl_uint));
	cl_uint* arguments = (cl_uint*)_alloca(number_arguments * sizeof(cl_uint));
	size_t i = 0;
	for (; i < number_images; i++)
		index_kernel_buffer[i] = i;
	for (size_t j = 0; i < number_images + number_arguments; i++) {
		index_kernel_arguments[j++] = i;
	}
	i = 0;
	for (; i < number_arguments; i++)
		arguments[i] = indices_arguments[i];
	size_t work_size[] = { indices_arguments[0], indices_arguments[1], 1 };
	device->setArguments(index_kernel, NULL, NULL, indices_images, number_images, index_kernel_buffer, arguments, type_arguments, number_arguments, index_kernel_arguments);
	device->startCalculate(index_kernel, work_size);
	length_last_buffer = number_images;
	last_buffers_modify = (cl_uint*)realloc(last_buffers_modify, length_last_buffer * sizeof(cl_uint));
	for (size_t j = 0; j < length_last_buffer; j++) {
		last_buffers_modify[j] = indices_images[j];
	}
}
void Lab_2::convert_float4_to_uchar4_image(size_t index_kernel, clDevice* device, size_t width, size_t height, cl_uint indices_images[2]) {
	cl_uchar type_arguments[] = { sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint) };
	cl_uint index_kernel_buffer[] = { 0, 1, };
	cl_uint index_kernel_arguments[] = { 2, 3 };
	size_t work_size[] = { width, height, 1 };
	cl_uint arguments[] = { width,  height };
	device->setArguments(index_kernel, NULL, NULL, indices_images, 2, index_kernel_buffer, arguments, type_arguments, 2, index_kernel_arguments);
	device->startCalculate(index_kernel, work_size);
	length_last_buffer = 2;
	last_buffers_modify = (cl_uint*)realloc(last_buffers_modify, length_last_buffer * sizeof(cl_uint));
	for (size_t j = 0; j < length_last_buffer; j++) {
		last_buffers_modify[j] = indices_images[j];
	}
}
void Lab_2::convolution_rgba_device(size_t index_kernel, clDevice* device, size_t width, size_t height, size_t width_kernel, size_t height_kernel, cl_uint indices_images[3]) {
	cl_uchar type_arguments[] = { sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint) };
	cl_uint index_kernel_buffer[] = { 0, 1, 2 };
	cl_uint index_kernel_arguments[] = { 3, 4, 5, 6 };
	size_t work_size[] = { width, height, 1 };
	cl_uint arguments[] = { width,  height , width_kernel, height_kernel };
	device->setArguments(index_kernel, NULL, NULL, indices_images, 3, index_kernel_buffer, arguments, type_arguments, 4, index_kernel_arguments);
	device->startCalculate(index_kernel, work_size);
	length_last_buffer = 3;
	last_buffers_modify = (cl_uint*)realloc(last_buffers_modify, length_last_buffer * sizeof(cl_uint));
	for (size_t j = 0; j < 3; j++) {
		last_buffers_modify[j] = indices_images[j];
	}
}

void Lab_2::make_noise_image_rgba_device(size_t index_kernel, clDevice* device, size_t width, size_t height, cl_uint indices_images[2]) {
	cl_uchar type_arguments[] = { sizeof(cl_uint),  sizeof(cl_uint) };
	cl_uint index_kernel_buffer[] = { 0, 1 };
	cl_uint index_kernel_arguments[] = { 2, 3 };
	size_t work_size[] = { width, height, 1 };
	cl_uint arguments[] = { width,  height };
	device->setArguments(index_kernel, NULL, NULL, indices_images, 2, index_kernel_buffer, arguments, type_arguments, sizeof(index_kernel_arguments) / sizeof(cl_uint), index_kernel_arguments);
	device->startCalculate(index_kernel, work_size);
	length_last_buffer = 2;
	last_buffers_modify = (cl_uint*)realloc(last_buffers_modify, length_last_buffer * sizeof(cl_uint));
	last_buffers_modify[0] = indices_images[0];
	last_buffers_modify[1] = indices_images[1];
}

void Lab_2::divide_fourier_transform_rgba_image_device(size_t index_kernel, clDevice* device, size_t width, size_t height, size_t width_kernel, size_t height_kernel, cl_uint indices_images[6]) {
	cl_uchar type_arguments[] = { sizeof(cl_uint),  sizeof(cl_uint), sizeof(cl_uint),  sizeof(cl_uint) };
	cl_uint index_kernel_buffer[] = { 0, 1, 2, 3, 4, 5 };
	cl_uint index_kernel_arguments[] = { 6, 7, 8, 9 };
	size_t work_size[] = { width, height, 1 };
	cl_uint arguments[] = { width,  height, width_kernel, height_kernel };
	device->setArguments(index_kernel, NULL, NULL, indices_images, 6, index_kernel_buffer, arguments, type_arguments, sizeof(index_kernel_arguments) / sizeof(cl_uint), index_kernel_arguments);
	device->startCalculate(index_kernel, work_size);
	length_last_buffer = 6;
	last_buffers_modify = (cl_uint*)realloc(last_buffers_modify, length_last_buffer * sizeof(cl_uint));
	for (size_t i = 0; i < length_last_buffer; i++)
		last_buffers_modify[i] = indices_images[i];
}

void Lab_2::fourier_transform_rgba_image_device(size_t index_kernel, clDevice* device, size_t width, size_t height, cl_uint indices_images[3]) {
	cl_uchar type_arguments[] = { sizeof(cl_uint),  sizeof(cl_uint) };
	cl_uint index_kernel_buffer[] = { 0, 1, 2 };
	cl_uint index_kernel_arguments[] = { 3, 4 };
	size_t work_size[] = { width, height, 1 };
	cl_uint arguments[] = { width,  height };
	device->setArguments(index_kernel, NULL, NULL, indices_images, 3, index_kernel_buffer, arguments, type_arguments, sizeof(index_kernel_arguments) / sizeof(cl_uint), index_kernel_arguments);
	device->startCalculate(index_kernel, work_size);
	length_last_buffer = 3;
	last_buffers_modify = (cl_uint*)realloc(last_buffers_modify, length_last_buffer * sizeof(cl_uint));
	last_buffers_modify[0] = indices_images[0];
	last_buffers_modify[1] = indices_images[1];
	last_buffers_modify[2] = indices_images[2];
}

void Lab_2::inverse_fourier_transform_rgba_image_device(size_t index_kernel, clDevice* device, size_t width, size_t height, cl_uint indices_images[3]) {
	cl_uchar type_arguments[] = { sizeof(cl_uint),  sizeof(cl_uint) };
	cl_uint index_kernel_buffer[] = { 0, 1, 2 };
	cl_uint index_kernel_arguments[] = { 3, 4 };
	size_t work_size[] = { width, height, 1 };
	cl_uint arguments[] = { width,  height };
	device->setArguments(index_kernel, NULL, NULL, indices_images, 3, index_kernel_buffer, arguments, type_arguments, sizeof(index_kernel_arguments) / sizeof(cl_uint), index_kernel_arguments);
	device->startCalculate(index_kernel, work_size);
	length_last_buffer = 3;
	last_buffers_modify = (cl_uint*)realloc(last_buffers_modify, length_last_buffer * sizeof(cl_uint));
	last_buffers_modify[0] = indices_images[0];
	last_buffers_modify[1] = indices_images[1];
	last_buffers_modify[2] = indices_images[2];
}



size_t Lab_2::mallocMemoryImage(clDevice* device, const void* ptr, size_t width, size_t height, size_t type_data, size_t type_image) {
	size_t _type_image[] = { type_image };
	size_t _type_data[] = { type_data };
	size_t length_row_pitch_data[1];
	size_t length_height_data[] = { height };
	size_t length_width_data[] = { width };
	if (type_data == CL_FLOAT && type_image == CL_RGBA)
		length_row_pitch_data[0] = { width * sizeof(cl_float4) };
	if (type_data == CL_UNORM_INT8 && type_image == CL_RGBA)
		length_row_pitch_data[0] = { width * sizeof(cl_uchar4) };
	else if (type_data == CL_UNSIGNED_INT8 && type_image == CL_RGBA)
		length_row_pitch_data[0] = { width * sizeof(cl_uchar4) };
	size_t _indices = device->mallocImageMemory(&ptr, length_height_data, length_width_data, length_row_pitch_data, 1, &_type_image[0], &_type_data[0]);
	const char* buffers[1] = { "image2d" + 0 };
	const size_t length_buffers[] = { 9 };
	number_buffers += 1;
	length_last_buffer = 1;
	types_buffers_OpenCL = (cl_uchar**)realloc(types_buffers_OpenCL, number_buffers * sizeof(cl_uchar*));
	length_buffer_OpenCL = (size_t*)realloc(length_buffer_OpenCL, number_buffers * sizeof(size_t));
	indices_buffer_OpenCL = (size_t*)realloc(indices_buffer_OpenCL, number_buffers * sizeof(size_t));
	last_buffers_modify = (cl_uint*)realloc(last_buffers_modify, length_last_buffer * sizeof(cl_uint));
	for (size_t i = number_buffers - 1, j = 0; i < number_buffers; i++, j++) {
		types_buffers_OpenCL[i] = (cl_uchar*)malloc(length_buffers[j] * sizeof(char));
		memcpy(types_buffers_OpenCL[i], buffers[j], length_buffers[j] * sizeof(char));
		indices_buffer_OpenCL[i] = _indices + j;
		length_buffer_OpenCL[i] = length_buffers[j];
		last_buffers_modify[j] = i;
	}
	return _indices;
}
void Lab_2::readImage(clDevice* device, void* returnData, size_t width_image, size_t height_image, cl_uint index_buffer) {
	cl_uchar type_arguments[] = { sizeof(cl_uint) };
	device->readImage((void**)&returnData, &index_buffer, type_arguments, &width_image, &height_image, 1);
}
Lab_2::Lab_2(clDevice* device, cl_uchar4* image, size_t width, size_t height, cl_float4* kernel, size_t width_kernel, size_t height_kernel)
{

	for (size_t i = 0; i < sizeof(*this); i++)
		((cl_char*)this)[i] = 0;
	const cl_char convolution_image[] = "convolution_image_rgba";
	const cl_char noise_image[] = "noise_image_rgba";
	const cl_char fourier_transform_image[] = "fourier_transform_rgba_image";
	const cl_char inverse_fourier_transform_image[] = "inverse_fourier_transform_rgba_image";
	const cl_char inverse_data_image_rgba[] = "inverse_data_image_rgba";
	const cl_char div_fourier_image_rgba_image[] = "div_fourier_image_rgba_image";
	const cl_char mul_fourier_image_rgba_image[] = "mul_fourier_image_rgba_image";
	const cl_char convert_float4_to_uchar4_rgba_image[] = "convert_float4_to_uchar4_image_rgba";
	const cl_char fourier_transform_float_rgba_image[] = "fourier_transform_float_rgba_image";
	cl_int magnitude_fourier = device->findKernel((const cl_char*)"fourier_magnitude_float4_to_uchar4_image_rgba", sizeof("fourier_magnitude_float4_to_uchar4_image_rgba"));
	cl_int phase_fourier = device->findKernel((const cl_char*)"fourier_phase_float4_to_uchar4_image_rgba", sizeof("fourier_phase_float4_to_uchar4_image_rgba"));
	//cl_int convolution_kernel_index = device->findKernel(convolution_image, sizeof("convolution_image_rgba"));
	cl_int convolution_kernel_index = device->findKernel((const cl_char*)"convolution_f_image_rgba", sizeof("convolution_f_image_rgba"));
	cl_int noise_kernel_index = device->findKernel(noise_image, sizeof("noise_image_rgba"));
	cl_int fourier_transform_kernel_index = device->findKernel(fourier_transform_image, sizeof("fourier_transform_rgba_image"));
	cl_int inverse_fourier_transform_kernel_index = device->findKernel(inverse_fourier_transform_image, sizeof("inverse_fourier_transform_rgba_image"));
	cl_int div_fourier_kernel_index = device->findKernel(div_fourier_image_rgba_image, sizeof("div_fourier_image_rgba_image"));
	cl_int mul_fourier_kernel_index = device->findKernel(mul_fourier_image_rgba_image, sizeof("mul_fourier_image_rgba_image"));
	cl_int convert_float4_to_uchar4_kernel_index = device->findKernel(convert_float4_to_uchar4_rgba_image, sizeof("convert_float4_to_uchar4_image_rgba"));
	cl_int fourier_transform_float4_kernel_index = device->findKernel(fourier_transform_float_rgba_image, sizeof("fourier_transform_float_rgba_image"));

	size_t real_data_gpu = mallocMemoryImage(device, NULL, width, height, CL_FLOAT, CL_RGBA);
	size_t imagine_data_gpu = mallocMemoryImage(device, NULL, width, height, CL_FLOAT, CL_RGBA);
	size_t real_kernel_gpu = mallocMemoryImage(device, NULL, width, height, CL_FLOAT, CL_RGBA);
	size_t imagine_kernel_gpu = mallocMemoryImage(device, NULL, width, height, CL_FLOAT, CL_RGBA);
	size_t real_result_image_gpu = mallocMemoryImage(device, NULL, width, height, CL_FLOAT, CL_RGBA);
	size_t imagine_result_image_gpu = mallocMemoryImage(device, NULL, width, height, CL_FLOAT, CL_RGBA);
	size_t kernel_gpu = mallocMemoryImage(device, kernel, width_kernel, height_kernel, CL_FLOAT, CL_RGBA);
	size_t image_gpu = mallocMemoryImage(device, image, width, height, CL_UNORM_INT8, CL_RGBA);
	size_t result_image_gpu = mallocMemoryImage(device, NULL, width, height, CL_UNORM_INT8, CL_RGBA);
	cl_uint indices[6] = { image_gpu, kernel_gpu, result_image_gpu };
	cl_uint indices_args[6] = { width, height, width_kernel, height_kernel };
	convolution_rgba_device(convolution_kernel_index, device, width, height, width_kernel, height_kernel, indices);

	indices[0] = result_image_gpu;
	indices[1] = image_gpu;

	make_noise_image_rgba_device(noise_kernel_index, device, width, height, indices);
	
	indices[0] = image_gpu;
	indices[1] = kernel_gpu;
	indices[2] = result_image_gpu;
	//convolution_rgba_device(convolution_kernel_index, device, width, height, width_kernel, height_kernel, indices);
	indices[0] = image_gpu;
	indices[1] = real_data_gpu;
	indices[2] = imagine_data_gpu;
	fourier_transform_rgba_image_device(fourier_transform_float4_kernel_index, device, width, height, indices);
	indices[0] = kernel_gpu;
	indices[1] = real_kernel_gpu;
	indices[2] = imagine_kernel_gpu;
	fourier_transform_rgba_image_device(fourier_transform_float4_kernel_index, device, width, height, indices);
	indices[0] = real_data_gpu;
	indices[1] = imagine_data_gpu;
	indices[2] = real_kernel_gpu;
	indices[3] = imagine_kernel_gpu;
	indices[4] = real_result_image_gpu;
	indices[5] = imagine_result_image_gpu;
	divide_fourier_transform_rgba_image_device(div_fourier_kernel_index, device, width, height, width_kernel, height_kernel, indices);
	indices[0] = real_result_image_gpu;
	indices[1] = imagine_result_image_gpu;
	indices[2] = result_image_gpu;
	inverse_fourier_transform_rgba_image_device(inverse_fourier_transform_kernel_index, device, width, height, indices);
	//indices[0] = real_kernel_gpu;
	//indices[1] = imagine_kernel_gpu;
	//indices[2] = result_image_gpu; 
	//call_opencl_function(phase_fourier, device, indices, indices_args, 3, 2);
	//convert_float4_to_uchar4_image(convert_float4_to_uchar4_kernel_index, device, width, height, indices);

	readImage(device, image, width, height, result_image_gpu);

	free(kernel);
	//free(inverse_kernel);
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