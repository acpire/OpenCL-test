
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

__kernel void convolution_rgba(__global uchar4* image,const int width,const int height, __global uchar4* kernel_convolution,const int width_kernel,const int height_kernel, __global uchar4* result)
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
						sum += index_y >= 0 && index_y < height && index_x >= 0 && index_x < width ? convert_float4(image[index_y * width + index_x]) * convert_float4(kernel_convolution[i * width_kernel + j]) : 0.0f;
					}
				}
				result[h * width +w] = convert_uchar4(sum / operations);
			}
		}
}
__kernel void noise_rgba(__global uchar4* image,const int width,const int height, float mathematical_expectation, float standard_deviation  )
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
	float math_exp = mathematical_expectation;
	float stand_dev = standard_deviation;
	float max =  (1.0f / (stand_dev * sqrt(2.0f * 3.1415926535f))) * exp(-pow(0.0f - math_exp, 2) / 2.0f * stand_dev * stand_dev);
	float min =  (1.0f / (stand_dev * sqrt(2.0f * 3.1415926535f))) * exp(-pow(1.0f - math_exp, 2) / 2.0f * stand_dev * stand_dev);
	float normalize = 255.0f / (max - min);
//	float step = 1.0f / float(width * height); 
	size_t end = height * width;
	for (size_t i = idy * stride_x+ idx; i < end ; i += stride_y * stride_x){
		uint seed = (i * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
		float index = 1.0f / seed;
		float part_one =  (1.0f / (stand_dev * sqrt(2.0f * 3.1415926535f)));
		float tmp = (index - math_exp);
		float part_two = ( tmp * tmp ) / (2.0f * stand_dev * stand_dev);
		float part_three = native_exp(-part_two);
		float result = part_one * part_three;
		float4 noise = result * 255.0f;
	//	noise -= min;
	//	noise *= normalize;
		image[i] += convert_uchar4(noise);
	}
}

)==="