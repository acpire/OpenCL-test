R"===(

__kernel void inverse_matrix_Gaussian_filter_float(read_only image2d_t image, write_only image2d_t image_write, const int width, const int height){

    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);
	

	int2 coord_xy;
    float rad = (3.1415926535f / 180.0f) * 180.0f;
	float2 center_rotate_xy;

	for (int h = idy; h < height ; h += stride_y){
		for (int w = idx ; w <  width ; w += stride_x){
			float4 data_image = read_imagef(image, (int2)(w, h));
			if ( w < width / 2 && h < height / 2){
				 center_rotate_xy = (float2)(width / 4, height / 4);
			}else if (w > width / 2 && h < height / 2){
				 center_rotate_xy = (float2)(3 * width / 4, height / 4);
			}else if (w < width / 2 && h > height / 2){
				 center_rotate_xy = (float2)(width / 4,3 * height / 4);
			}else if (w > width / 2 && h > height / 2){
				 center_rotate_xy = (float2)(3 * width / 4,3 * height / 4);
			}
			coord_xy.x = center_rotate_xy.x + (w - center_rotate_xy.x) * native_cos(rad) - (h - center_rotate_xy.y) * native_sin(rad);
			coord_xy.y = center_rotate_xy.y + (w - center_rotate_xy.x) * native_sin(rad) + (h - center_rotate_xy.y) * native_cos(rad);

			write_imagef(image_write, coord_xy, data_image);


		}
	}




}



//__kernel void WienerFiltration_float_image(read_only image2d Re_image, read_only image2d Im_image, read_only image2d Re_kernel, read_only image2d Im_kernel, )







)==="