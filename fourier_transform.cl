R"===(

__kernel void fourier_transform_float_rgba_image(read_only image2d_t image_read, write_only image2d_t image_write_Re, write_only image2d_t image_write_Im, const int width, const int height )
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
    const float pi2 = 2.0f * 3.14159265359f;
	const float f_height = 1.0f / convert_float(height);
	const float f_width = 1.0f / convert_float(width);
	float normalization = width * height;
	float4 Re_sum;
	float4 Im_sum;
	for (int y = idy; y < height; y+=stride_y) {
		for (int x = idx; x < width; x+=stride_x) {
			Re_sum = 0.0f;
			Im_sum = 0.0f;
			for (int i = 0 ; i < height; i++){
				for (int j = 0; j < width; j++){
					const float x_angle = x  * convert_float(j);
					const float y_angle = y  * convert_float(i);
					const float index = pi2 * (x_angle * f_width + y_angle * f_height) ;
					float4 data_image = 255.0f * read_imagef(image_read, (int2)(j, i));
					float4 Re_data = data_image * native_cos(index);
					float4 Im_data = data_image * native_sin(index);
					Re_sum += Re_data;
					Im_sum -= Im_data;
				}
			}
			//if (y < 1 && x < 60)
			//printf("%v4f \n", Re_sum);
			write_imagef(image_write_Re, (int2)(x ,  y), (float4)(Re_sum));
			write_imagef(image_write_Im, (int2)(x ,  y), (float4)(Im_sum));
		}
	}
}

__kernel void fourier_transform_rgba_image(read_only image2d_t image_read, write_only image2d_t image_write_Re, write_only image2d_t image_write_Im, const int width, const int height )
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
    const float pi2 = 2.0f * 3.14159265359f;
	const float f_height = 1.0f / convert_float(height);
	const float f_width = 1.0f / convert_float(width);
	float normalization = width * height;
	float4 Re_sum;
	float4 Im_sum;
	for (int y = idy; y < height; y+=stride_y) {
		for (int x = idx; x < width; x+=stride_x) {
			Re_sum = 0.0f;
			Im_sum = 0.0f;
			for (int i = 0 ; i < height; i++){
				for (int j = 0; j < width; j++){
					const float x_angle = x  * convert_float(j);
					const float y_angle = y  * convert_float(i);
					const float index = pi2 * (x_angle * f_width + y_angle * f_height) ;
					float4 data_image = convert_float4(read_imageui(image_read, (int2)(j, i)));
					float4 Re_data = data_image * native_cos(index);
					float4 Im_data = data_image * native_sin(index);
					Re_sum += Re_data;
					Im_sum -= Im_data;
				}
			}
			write_imagef(image_write_Re, (int2)(x ,  y), (float4)(Re_sum));
			write_imagef(image_write_Im, (int2)(x ,  y), (float4)(Im_sum));
		}
	}
}

__kernel void mul_fourier_image_rgba_image(read_only image2d_t image_read_Re, read_only image2d_t image_read_Im, read_only image2d_t kernel_read_Re, read_only image2d_t kernel_read_Im, write_only image2d_t image_write_Re, write_only image2d_t image_write_Im, const int width, const int height, const int width_kernel, const int height_kernel )
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);

	for (int i = idy; i < height; i+=stride_y) {
		for (int j = idx; j < width; j+=stride_x) {
			float4 Re_data_image = read_imagef(image_read_Re, (int2)(j, i));
			float4 Im_data_image = read_imagef(image_read_Im, (int2)(j, i));
			float4 Re_data_kernel = read_imagef(kernel_read_Re, (int2)(j, i));
			float4 Im_data_kernel = read_imagef(kernel_read_Im, (int2)(j, i));
			float4 Re_new_image = (Re_data_image * Re_data_kernel - Im_data_image * Im_data_kernel) ;
			float4 Im_new_image = (Im_data_image * Re_data_kernel + Re_data_image * Im_data_kernel) ;
			write_imagef(image_write_Re, (int2)(j ,  i), Re_new_image);
			write_imagef(image_write_Im, (int2)(j ,  i), Im_new_image);
			
		}
	}
}

__kernel void div_fourier_image_rgba_image(read_only image2d_t image_read_Re, read_only image2d_t image_read_Im, read_only image2d_t kernel_read_Re, read_only image2d_t kernel_read_Im, write_only image2d_t image_write_Re, write_only image2d_t image_write_Im, const int width, const int height, const int width_kernel, const int height_kernel )
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
	for (int i = idy; i < height; i+=stride_y) {
		for (int j = idx; j < width; j+=stride_x) {
			float4 Re_data_image = read_imagef(image_read_Re, (int2)(j, i));
			float4 Im_data_image = read_imagef(image_read_Im, (int2)(j, i));
			float4 Re_data_kernel = read_imagef(kernel_read_Re, (int2)(j, i));
			float4 Im_data_kernel = read_imagef(kernel_read_Im, (int2)(j, i));
			
			//float4 Re_new_image = Re_data_image / Re_data_kernel;
			//float4 Im_new_image = Im_data_image / Im_data_kernel;
			const float4 div = 1.0f / (Re_data_kernel * Re_data_kernel + Im_data_kernel * Im_data_kernel );
			//const float4 div  = 1.0f / 500.0f;
			float4 Re_new_image = (Re_data_image * Re_data_kernel + Im_data_image * Im_data_kernel) * div;
			float4 Im_new_image = (Im_data_image * Re_data_kernel - Re_data_image * Im_data_kernel) * div;
			write_imagef(image_write_Re, (int2)(j ,  i), Re_new_image );
			write_imagef(image_write_Im, (int2)(j ,  i), Im_new_image );
			
		}
	}
}

__kernel void inverse_fourier_transform_rgba_image(read_only image2d_t image_read_Re, read_only image2d_t image_read_Im, write_only image2d_t image_write, const int width, const int height )
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
    const float pi2 = 2.0f * 3.14159265359f;
	const float f_height = 1.0f / convert_float(height);
	const float f_width = 1.0f / convert_float(width);
	float normalization = width * height;
	float4 sum;
	for (int y = idy; y < height; y+=stride_y) {
		for (int x = idx; x < width; x+=stride_x) {
			sum = 0.0f;
			for (int i = 0 ; i < height; i++){
				for (int j = 0; j < width; j++){
					const float index = pi2 * (x *  f_width * convert_float(j) +  y * f_height * convert_float(i)) ;
					float4 Re_data = read_imagef(image_read_Re, (int2)(j, i));
					float4 Im_data = read_imagef(image_read_Im, (int2)(j, i));
					sum += Re_data * native_cos(index) - Im_data * native_sin(index);
				}
			}
			write_imagef(image_write, (int2)(x ,  y), sum/normalization);
		}
	}
}


__kernel void fourier_transform_rgba(__global uchar4* image,__global uchar4* result, const int width,const int height )
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
    float minus_pi2 = -2.0f * 3.14159265359f;
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
					float4 data_image = convert_float4(image[h_offset + j]);
					sum += (data_image * native_exp(minus_pi2 * (index_y * f_height + x * convert_float(j) * f_width)));
					j++;
				}
			}
			result[y * width + x] = convert_uchar4(  sum  );
		}
	}
}

__kernel void fourier_magnitude_float4_image_rgba(read_only image2d_t real_image, read_only image2d_t im_image, write_only image2d_t  image_write,const int width,const int height)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
	for (int h = idy; h < height ; h += stride_y){
		for (int w = idx ; w <  width ; w += stride_x){
			float4 data_image_re = read_imagef(real_image, (int2)(w, h));
			float4 data_image_im = read_imagef(im_image, (int2)(w, h));
			float4 result = sqrt(data_image_re*data_image_re+data_image_im*data_image_im);
			write_imagef(image_write, (int2)(w ,  h), (result));
		}
	}
}

__kernel void fourier_phase_float4_image_rgba(read_only image2d_t real_image, read_only image2d_t im_image, write_only image2d_t  image_write,const int width,const int height)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
	for (int h = idy; h < height ; h += stride_y){
		for (int w = idx ; w <  width ; w += stride_x){
			float4 data_image_re = read_imagef(real_image, (int2)(w, h));
			float4 data_image_im = read_imagef(im_image, (int2)(w, h));
			float4 result = (atan2(data_image_im, data_image_re) + 3.1415926535897932384f) / (2.0f * 3.1415926535897932384f);
			write_imagef(image_write, (int2)(w ,  h), (result));
		}
	}
}

__kernel void mul_float4_image_rgba(read_only image2d_t data_image_1, read_only image2d_t data_image_2, write_only image2d_t  image_write,const int width,const int height)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
	for (int h = idy; h < height ; h += stride_y){
		for (int w = idx ; w <  width ; w += stride_x){
			float4 data_image_re = read_imagef(data_image_1, (int2)(w, h));
			float4 data_image_im = read_imagef(data_image_2, (int2)(w, h));
			float4 result = data_image_re * data_image_im;
			printf("%f \n", result.x);
			write_imagef(image_write, (int2)(w ,  h), (result));
		}
	}
}





)==="