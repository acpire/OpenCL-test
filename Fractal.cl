R"===(


__kernel void fractal_float_image_rgba(read_only image2d_t image, write_only image2d_t  result_image, write_only image2d_t  compress_information, write_only image2d_t  compress_type, const int width,const int height, const int size_rank){


    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int stride_x = get_global_size(0);
    const int stride_y = get_global_size(1);




}







)==="