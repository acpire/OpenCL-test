R"===(

#pragma OPENCL EXTENSION cl_amd_printf: enable  
static float noise3D(float x, float y, float z) {
    float ptr = 0.0f;
	return fract(sin(x*112.9898f + y * 179.233f + z * 237.212f) * 43758.5453f, &ptr);
}
//static float erf(float x) {
//	return 1.0f - (1.0f)/pow(1.0f + 0.278393f*x + 0.230389*x*x+0.000972f*x*x*x+0.078108f*x*x*x*x, 4);
//}
__kernel void noise_image_rgba(read_only image2d_t read_image, write_only image2d_t write_image, const int width,const int height  )
{

    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);

	float math = 0.0f;
	float D = 0.2f;
	for (size_t y = idy; y < height; y+=stride_y) {
		for (size_t x = idx; x < width; x+=stride_x) {
			float vector_x  = convert_float(x+1);
			float vector_y  = convert_float(y+1);
			float vector_z  = convert_float((x*y)+1);
			float random_number_x = noise3D(vector_x, vector_y, vector_z);
			float random_number_y = noise3D(vector_y, vector_z, vector_x);
			float random_number_z = noise3D(vector_z, vector_x, vector_y);
			float4 random_number = 1.0f - 2.0f * (float4)(random_number_x, random_number_y, random_number_z, 0.0f);
			random_number =  (erf((random_number - math) / sqrt(2.0f * D))) ;
			float4 data_image = read_imagef(read_image, (int2)(x ,  y));
			write_imagef(write_image, (int2)(x ,  y), (data_image + random_number));
		}
	}
}

__kernel void convert_float4_to_uchar4_image_rgba(read_only image2d_t image, write_only image2d_t  image_write,const int width,const int height)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
	for (int h = idy; h < height ; h += stride_y){
		for (int w = idx ; w <  width ; w += stride_x){
			float4 data_image = read_imagef(image, (int2)(w, h));
			write_imageui(image_write, (int2)(w ,  h), convert_uint4(data_image));
		}
	}
}

__kernel void convert_uchar4_to_float4_image_rgba(read_only image2d_t image, write_only image2d_t  image_write,const int width,const int height)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
	for (int h = idy; h < height ; h += stride_y){
		for (int w = idx ; w <  width ; w += stride_x){
			uint4 data_image = read_imageui(image, (int2)(w, h));
			write_imagef(image_write, (int2)(w ,  h), convert_float4(data_image));
		}
	}
}
__kernel void convolution_f_image_rgba(read_only image2d_t image, read_only image2d_t  kernel_convolution, write_only image2d_t  image_write,const int width,const int height,const int width_kernel,const int height_kernel)
{
	const sampler_t smp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE |  CLK_FILTER_NEAREST; 

	int2 int_index_xy = (int2)(get_global_id(0),  get_global_id(1));
	const int2 int_stride_xy = (int2)(get_global_size(0), get_global_size(1));
    const float idx = convert_float(int_index_xy.x)/convert_float(width);
    const float idy = convert_float(int_index_xy.y)/convert_float(height);

    const float stride_x = convert_float(int_stride_xy.x)/convert_float(width);
    const float stride_y = convert_float(int_stride_xy.y)/convert_float(height);

    const float step_kernel_x = 1.0f/convert_float(width_kernel);
    const float step_kernel_y = 1.0f/convert_float(height_kernel);
    const float step_image_x = 1.0f/convert_float(width);
    const float step_image_y = 1.0f/convert_float(height);

   float part_size_kernel_x = -(convert_float(width_kernel) * 0.5f)*step_image_x;
   float part_size_kernel_y = -(convert_float(height_kernel) * 0.5f)*step_image_y;

	const float4 operations = 1.0f / (height_kernel * width_kernel ) ;
	for (float h = idy; h < 1.0f ; h += stride_y){
		for (float w = idx ; w <  1.0f ; w += stride_x){
			float4 sum = 0.0f;
			float index_image_y = h + part_size_kernel_y;
			for (float i = 0.0f; i < 1.0f; i+=step_kernel_y){
				float index_image_x = w + part_size_kernel_x;
				for (float j = 0.0f; j < 1.0f; j+=step_kernel_x){
					const float4 data_image = read_imagef(image, smp, (float2)(index_image_x, index_image_y));
					
					const float4 data_kernel = read_imagef(kernel_convolution, smp, (float2)(j, i));
					sum += data_image * data_kernel;
					index_image_x += step_image_x;
				}
				index_image_y += step_image_y;
			}
			sum *= operations;
			//float4 data_image =read_imagef(image, smp, (float2)(w, h));
			//printf("%f\n" data_image.x );
			write_imagef(image_write, int_index_xy, sum);
			int_index_xy += int_stride_xy;
		}
	}
}
__kernel void convolution_image_rgba(read_only image2d_t image, read_only image2d_t  kernel_convolution, write_only image2d_t  image_write,const int width,const int height,const int width_kernel,const int height_kernel)
{
	const sampler_t smp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE |  CLK_FILTER_NEAREST; 

	int2 int_index_xy = (int2)(get_global_id(0),  get_global_id(1));
	const int2 int_stride_xy = (int2)(get_global_size(0), get_global_size(1));
    const float idx = convert_float(int_index_xy.x)/convert_float(width);
    const float idy = convert_float(int_index_xy.y)/convert_float(height);

    const float stride_x = convert_float(int_stride_xy.x)/convert_float(width);
    const float stride_y = convert_float(int_stride_xy.y)/convert_float(height);

    const float step_kernel_x = 1.0f/convert_float(width_kernel);
    const float step_kernel_y = 1.0f/convert_float(height_kernel);
    const float step_image_x = 1.0f/convert_float(width);
    const float step_image_y = 1.0f/convert_float(height);

   float part_size_kernel_x = -(convert_float(width_kernel) * 0.5f)*step_image_x;
   float part_size_kernel_y = -(convert_float(height_kernel) * 0.5f)*step_image_y;

	const float4 operations = 1.0f / (height_kernel * width_kernel ) ;
	for (float h = idy; h < 1.0f ; h += stride_y){
		for (float w = idx ; w <  1.0f ; w += stride_x){
			float4 sum = 0.0f;
			float index_image_y = h + part_size_kernel_y;
			for (float i = 0.0f; i < 1.0f; i+=step_kernel_y){
				float index_image_x = w + part_size_kernel_x;
				for (float j = 0.0f; j < 1.0f; j+=step_kernel_x){
					const float4 data_image = convert_float4(read_imageui(image, smp, (float2)(index_image_x, index_image_y)));
					const float4 data_kernel = read_imagef(kernel_convolution, smp, (float2)(j, i));
					sum += data_image * data_kernel;
					index_image_x += step_image_x;
				}
				index_image_y += step_image_y;
			}
			sum *= operations;
			write_imageui(image_write, int_index_xy, convert_uint4(sum));
			int_index_xy += int_stride_xy;
		}
	}
}
__kernel void convolution_image_with_local_memory_rgba(read_only image2d_t image, read_only image2d_t  kernel_convolution, write_only image2d_t  image_write,const int width,const int height, const int width_kernel,const int height_kernel)
{
	__local uchar4 matrix_kernel[256];
	__local uchar4 matrix_image[256];
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
	const int local_id_xy = get_local_id(0) + get_local_size(0) * get_local_id(1);
	const int4 part_width_height_kernel = -(int4)(width_kernel >> 1, height_kernel >> 1, 0, 0) ;
	const float operations = 1.0f/convert_float(height_kernel * width_kernel);
	for (int h = idy; h < height ; h += stride_y){
		for (int w = idx ; w <  width ; w += stride_x){
			float4 sum = 0.0f;
			int4 index_image_kernel = (int4)(w, h, 0, 0 ) + part_width_height_kernel;
			for (int i = 0; i < height_kernel; i+=16){
				for (int j = 0; j < width_kernel; j+=16){
					const int4 _index_image_kernel = index_image_kernel + (int4)(j, i, j, i );
					matrix_image[local_id_xy] = convert_uchar4(read_imageui(image, _index_image_kernel.s01));
					matrix_kernel[local_id_xy] = convert_uchar4(read_imageui(kernel_convolution, _index_image_kernel.s23));
					barrier(CLK_LOCAL_MEM_FENCE);
					for (int i = 0; i < 256; i++){
							sum += convert_float4(matrix_image[i]) *  convert_float4(matrix_kernel[i]);
						
					}
				}
			}
			sum *= operations;
			write_imageui(image_write, (int2)(w ,  h), convert_uint4(sum));
		}
	}
}


__kernel void convolution(__global uchar* image, __global uchar* kernel_convolution, __global uchar* result,const int width,const int height,const int width_kernel,const int height_kernel)
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

__kernel void convolution_rgba(__global uchar4* image, __global float4* kernel_convolution, __global uchar4* result,const int width,const int height, const int width_kernel,const int height_kernel)
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

)==="