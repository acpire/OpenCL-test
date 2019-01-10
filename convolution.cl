R"===(

__kernel void convolution(__global uchar* image,const int width,const int height, __global uchar* kernel_convolution,const int width_kernel,const int height_kernel, __global uchar* result)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
	const int part_height_kernel = height_kernel >> 1;
	const int part_width_kernel = width_kernel >> 1;
	const int operations = height_kernel * width_kernel;
	for (size_t h = idy + part_height_kernel; h < height - part_height_kernel; h+=stride_y){
		for (size_t w = idx + part_width_kernel; w <  width - part_width_kernel; w += stride_x){
			float sum = 0.0f;
			for (size_t i = 0; i < height_kernel; i++){
				for (size_t j = 0; j < width_kernel; j++){
						const int index_y = (h + i - part_height_kernel);
						const int index_x = (w + j - part_width_kernel);
						sum += index_y >= 0 && index_y < height && index_x >= 0 && index_x < width ? convert_float(image[index_y * width + index_x]) * convert_float(kernel_convolution[i * width_kernel + j]) : 0.0f;
				}
			}
			result[h * width +w] = convert_uchar(sum / operations);
		}
	}
}

__kernel void convolution_rgba(__global uchar4* image,const int width,const int height, __global float4* kernel_convolution,const int width_kernel,const int height_kernel, __global uchar4* result)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
	const int part_height_kernel = height_kernel >> 1;
	const int part_width_kernel = width_kernel >> 1;


	const float operations = convert_float(height_kernel * width_kernel);
		for (int h = idy; h < height ; h += stride_y){
			for (int w = idx ; w <  width ; w += stride_x){
				float4 sum = 0.0f;
				for (int i = 0; i < height_kernel; i++){
					for (int j = 0; j < width_kernel; j++){
						const int index_y = (h + i - part_height_kernel);
						const int index_x = (w + j - part_width_kernel);
						if (index_y >= 0 && index_y < height && index_x >= 0 && index_x < width){
							const float4 data_image = convert_float4(image[index_y * width + index_x]);
							sum += data_image * kernel_convolution[i * width_kernel + j];
						}
					}
				}
				sum /= operations;
				result[h * width +w] = convert_uchar4(sum);
			}
		}
}
__kernel void convolution_image_rgba(read_only image2d_t image,const int width,const int height, read_only image2d_t  kernel_convolution,const int width_kernel,const int height_kernel, write_only image2d_t  result)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
	const int part_height_kernel = height_kernel >> 1;
	const int part_width_kernel = width_kernel >> 1;


	const float operations = convert_float(height_kernel * width_kernel);
		for (int h = idy; h < height ; h += stride_y){
			for (int w = idx ; w <  width ; w += stride_x){
				float4 sum = 0.0f;
				for (int i = 0; i < height_kernel; i++){
					for (int j = 0; j < width_kernel; j++){
						const int2 index_image = (int2)((h + i - part_height_kernel) ,  (w + j - part_width_kernel));
						const int2 index_kernel = (int2)(i ,  j);
						if (index_image.y >= 0 && index_image.y < height && index_image.x >= 0 && index_image.x < width){
							const float4 data_image = convert_float4(read_imagei(image, index_image));
							sum += data_image * read_imagef(kernel_convolution, index_kernel);
						}
					}
				}
				sum /= operations;
				int2 index_write_image = (int2)(h ,  w);
				write_imageui(result, index_write_image, convert_uint4(sum));
			}
		}
}
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
__kernel void mathematical_expectation_rgba(__global uchar4* image, const int width,const int height,__global ulong* mathematical_expectation)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
	size_t end = height * width;
	for (size_t i = idy * stride_x+ idx; i < end ; i += stride_y * stride_x){
		ulong4 data = convert_ulong4(image[i]);
	    atom_add(&mathematical_expectation[0], data.s0);
	    atom_add(&mathematical_expectation[1], data.s1);
	    atom_add(&mathematical_expectation[2], data.s2);
	    atom_add(&mathematical_expectation[3], data.s3);
	}
	if (idx <1 && idy < 1){
		mathematical_expectation[0] /= end;
		mathematical_expectation[1] /= end;
		mathematical_expectation[2] /= end;
		mathematical_expectation[3] /= end;
	}
}
static float noise3D(float x, float y, float z) {
    float ptr = 0.0f;
	return fract(sin(x*112.9898f + y * 179.233f + z * 237.212f) * 43758.5453f, &ptr);
}
__kernel void noise_rgba(__global uchar4* image, const int width,const int height, float mathematical_expectation, float standard_deviation  )
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);

	float math = 3;
	float D = 1.0f;
	float C = 1.0f / (D * sqrt(2.0f * 3.14159265359f));
	float div = (2.0f * D * D);
	for (size_t y = idy; y < height; y+=stride_y) {
		for (size_t x = idx; x < width; x+=stride_x) {
			float random_number_x = noise3D(x, y, 0.0f);
			float random_number_y = noise3D(x, 0.0f, y);
			float random_number_z = noise3D(y, x, 0.0f);
			float4 random_number = (float4)(random_number_x, random_number_y, random_number_z, 0.0f);
			random_number *= 6.0f;
			random_number -= math;
			random_number *= random_number;
			random_number /= div;
			random_number = exp(-random_number);
			random_number *= C;
			random_number /= 0.4f;
			float4 data_image = convert_float4(image[y * width + x]);
			image[y * width + x] = convert_uchar4(data_image + random_number * 255.0f );
		}
	}

}
__kernel void fourier_transform_rgba_image(read_only image2d_t image,__global uchar4* result, const int width,const int height )
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
    float minus_pi2 = -2.0 * 3.14159265359f;
	float normalization = width * height;
	float f_height = 1.0f / convert_float(height);
	const float f_width = 1.0f / convert_float(width);
	for (int y = idy; y < height; y+=stride_y) {
		for (int x = idx; x < width; x+=stride_x) {
			float4 sum = 0.0f;
			for (int i = 0 ; i < height; i++){
				const float index_y = y * convert_float(i);
				int j = 0;
				while (j < width){
					int2 coord = (int2)(j, i);
					float4 data_image = read_imagef(image, coord);
					sum += (data_image * native_exp(minus_pi2 * (index_y * f_height + x * convert_float(j) * f_width)));
					j++;
				}
			}
			result[y * width + x] = convert_uchar4(  sum  );
		}
	}
}
__kernel void fourier_transform_rgba(__global uchar4* image,__global uchar4* result, const int width,const int height )
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
    float minus_pi2 = -2.0 * 3.14159265359f;
	float normalization = width * height;
	float f_height = 1.0f / convert_float(height);
	const float f_width = 1.0f / convert_float(width);
	for (int y = idy; y < height; y+=stride_y) {
		for (int x = idx; x < width; x+=stride_x) {
			float4 sum = 0.0f;
			for (int i = 0 ; i < height; i++){
				const float index_y = y * convert_float(i);
				const int h_offset = i * width;
				int j = 0;
				while (j < width){
					
					float4 data_image = convert_float(image[h_offset + j]);
					sum += (data_image * native_exp(minus_pi2 * (index_y * f_height + x * convert_float(j) * f_width)));
					j++;
				}
			}
			result[y * width + x] = convert_uchar4(  sum  );
		}
	}
}
//__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void fourier_transform_local_memory_rgba(__global uchar4* image,__global uchar4* result, const int width,const int height )
{
	__local float4 caching[64];
	const int local_index = get_group_id(1) *  get_local_size(0) + get_group_id(0);
	const int stride_local = get_local_size(0) * get_local_size(1);
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
    float minus_pi2 = -2.0 * 3.14159265359f;
	float normalization = width * height;
	float f_height = convert_float(height);
	float f_width = convert_float(width);
	const float x_number_strides = convert_float((width/64) + 1);
	for (int y = idy; y < height; y+=stride_y) {
		for (int x = idx; x < width; x+=stride_x) {
			float4 sum = 0.0f;
			for (int i = 0 ; i < height; i++){
				const float index_y = y * convert_float(i);
				for (float j = 0 ; j <  x_number_strides; j+=1.0f){
					for (int k = j * 64 + local_index ; k < 64 ; k += stride_local){
						if (k < width ){
							float4 tmp = convert_float4(image[i * width + k]);
							caching[k] = tmp;
						}else
							caching[k] =(float4)0;
					}
					barrier(CLK_LOCAL_MEM_FENCE);
					for (uchar k = 0; k < 64; k++){
						const float index_x = j * 64.0f + convert_float(k);
						float4 data_image = caching[k];
						sum += data_image * native_exp(minus_pi2 * (index_y / f_height + x * index_x / (f_width)));
					}
					barrier(CLK_LOCAL_MEM_FENCE);
				}
			}
			//if (idy == 1)
			//	printf("%v4f\n ", sum / normalization);
			//if (idy == 0)
			//   printf("%u\n ",   stride_local);
			result[y * width + x] = convert_uchar4( sum / normalization);
		}
	}

}
)==="