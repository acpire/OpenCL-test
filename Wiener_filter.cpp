#include "Wiener_filter.h"
void Wiener_filter::getSpectrum(cl_uint real_image_gpu, cl_uint imagine_image_gpu, cl_uint result_spectrum, cl_uint width, cl_uint height) {


	{
		cl_uint indices[] = { real_image_gpu, imagine_image_gpu, result_spectrum };
		cl_uint indices_args[] = { width, height };
		_device->callOpenclFunction(magnitude_fourier, indices, indices_args, 3, 2);
	}
	{
		cl_uint indices[] = { result_spectrum, result_spectrum, result_spectrum };
		cl_uint indices_args[] = { width, height };
		_device->callOpenclFunction(mul_float4_kernel_index, indices, indices_args, 3, 2);
	}



}
Wiener_filter::Wiener_filter(clDevice* device, cl_uchar4* image, size_t width, size_t height, cl_float4* kernel, size_t width_kernel, size_t height_kernel)
{
	const void* null_ptr = NULL;
	cl_uchar type_arguments[] = { sizeof(cl_uint) };
	_device = device;
	size_t _type_image[] = { CL_RGBA };
	size_t _type_data[] = { CL_FLOAT, CL_UNORM_INT8, CL_UNSIGNED_INT8 };
	size_t length_row_pitch_data[] = { width * sizeof(cl_float4), width * sizeof(cl_uchar4), width_kernel * sizeof(cl_float4), width_kernel * sizeof(cl_uchar4) };
	cl_int magnitude_fourier = device->findKernel((const cl_char*)"fourier_magnitude_float4_image_rgba", sizeof("fourier_magnitude_float4_image_rgba"));
	cl_int phase_fourier = device->findKernel((const cl_char*)"fourier_phase_float4_image_rgba", sizeof("fourier_phase_float4_image_rgba"));
	cl_int convolution_kernel_index = device->findKernel((const cl_char*)"convolution_f_image_rgba", sizeof("convolution_f_image_rgba"));
	cl_int noise_kernel_index = device->findKernel((const cl_char*)"noise_image_rgba", sizeof("noise_image_rgba"));
	cl_int fourier_transform_kernel_index = device->findKernel((const cl_char*)"fourier_transform_rgba_image", sizeof("fourier_transform_rgba_image"));
	cl_int inverse_fourier_transform_kernel_index = device->findKernel((const cl_char*)"inverse_fourier_transform_rgba_image", sizeof("inverse_fourier_transform_rgba_image"));
	cl_int div_fourier_kernel_index = device->findKernel((const cl_char*)"div_fourier_image_rgba_image", sizeof("div_fourier_image_rgba_image"));
	cl_int mul_fourier_kernel_index = device->findKernel((const cl_char*)"mul_fourier_image_rgba_image", sizeof("mul_fourier_image_rgba_image"));
	cl_int convert_float4_to_uchar4_kernel_index = device->findKernel((const cl_char*)"convert_float4_to_uchar4_image_rgba", sizeof("convert_float4_to_uchar4_image_rgba"));
	cl_int fourier_transform_float4_kernel_index = device->findKernel((const cl_char*)"fourier_transform_float_rgba_image", sizeof("fourier_transform_float_rgba_image"));
	cl_int invers_matrix_kernel_index = device->findKernel((const cl_char*)"inverse_matrix_Gaussian_filter_float", sizeof("inverse_matrix_Gaussian_filter_float"));
	cl_int mul_float4_kernel_index = device->findKernel((const cl_char*)"mul_float4_image_rgba", sizeof("mul_float4_image_rgba"));

	cl_uint image_gpu = device->mallocImageMemory((const void**)&image, &height, &width, length_row_pitch_data + 1, 1, _type_image, _type_data + 1);
	cl_uint result_image_gpu = device->mallocImageMemory(&null_ptr, &height, &width, length_row_pitch_data + 1, 1, _type_image, _type_data + 1);
	cl_uint real_image_gpu = device->mallocImageMemory(&null_ptr, &height, &width, length_row_pitch_data, 1, _type_image, _type_data);
	cl_uint imagine_image_gpu = device->mallocImageMemory(&null_ptr, &height, &width, length_row_pitch_data, 1, _type_image, _type_data);
	cl_uint spectrum_image_gpu = device->mallocImageMemory(&null_ptr, &height, &width, length_row_pitch_data, 1, _type_image, _type_data);
	cl_uint real_kernel_gpu = device->mallocImageMemory(&null_ptr, &height, &width, length_row_pitch_data, 1, _type_image, _type_data);
	cl_uint imagine_kernel_gpu = device->mallocImageMemory(&null_ptr, &height, &width, length_row_pitch_data, 1, _type_image, _type_data);
	cl_uint real_result_gpu = device->mallocImageMemory(&null_ptr, &height, &width, length_row_pitch_data, 1, _type_image, _type_data);
	cl_uint imagine_result_gpu = device->mallocImageMemory(&null_ptr, &height, &width, length_row_pitch_data, 1, _type_image, _type_data);
	cl_uint kernel_gpu = device->mallocImageMemory((const void**)&kernel, &height_kernel, &width_kernel, length_row_pitch_data + 2, 1, _type_image, _type_data);
	cl_uint copy_kernel_gpu = device->mallocImageMemory(&null_ptr, &height_kernel, &width_kernel, length_row_pitch_data + 2, 1, _type_image, _type_data);

	{
		cl_uint indices[] = { image_gpu, real_image_gpu, imagine_image_gpu };
		cl_uint indices_args[] = { width, height };
		device->callOpenclFunction(fourier_transform_float4_kernel_index, indices, indices_args, 3, 2);
	}
	getSpectrum(real_image_gpu, imagine_image_gpu, spectrum_image_gpu, width, height);
	{
		cl_uint indices[] = { real_image_gpu, imagine_image_gpu, image_gpu };
		cl_uint indices_args[] = { width, height, width_kernel, height_kernel };
		device->callOpenclFunction(inverse_fourier_transform_kernel_index, indices, indices_args, 3, 2);
	}

	device->readImage((void**)&image, &image_gpu, type_arguments, &width, &height, 1);
	device->freeImageMemory(real_image_gpu);
	device->freeImageMemory(imagine_image_gpu);
	device->freeImageMemory(real_kernel_gpu);
	device->freeImageMemory(imagine_kernel_gpu);
	device->freeImageMemory(real_result_gpu);
	device->freeImageMemory(imagine_result_gpu);
	device->freeImageMemory(kernel_gpu);
	device->freeImageMemory(image_gpu);
	device->freeImageMemory(result_image_gpu);
	device->freeImageMemory(copy_kernel_gpu);
	device->freeImageMemory(spectrum_image_gpu);
}

Wiener_filter::~Wiener_filter()
{
}