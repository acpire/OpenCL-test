R"===(
__kernel void convolution4(__global  uchar* image,  uint width,  uint height, __global  uchar* kernel,  uint widthKernel,  uint heightKernel){

		 size_t idx = get_global_id(0); 
		 size_t idy = get_global_id(1); 
		 size_t stride_x = get_global_size(0); 
		 size_t stride_y = get_global_size(1); 

		__global uchar4* ptrImage = image;
		__global uchar* ptrKernel = kernel;

		uint _width = width;
		uint residue = _width % 4;
		uint stepsWidthImage = _width >> 2;
		
	//	uint residue = widthKernel % 4;
	//	uint stepsWidthKernel = widthKernel >> 2;

		int partHeightKernel = heightKernel >> 1;
		int partWidthKernel = widthKernel >> 1;
		int startWidth = partWidthKernel >> 2;
		int endWidth = stepsWidthImage - (partWidthKernel >> 2) - 1;

		float4 readKernel;
		float readImage;
		float operations = heightKernel * widthKernel;

		for (uint h = partHeightKernel + idy; h < height - partHeightKernel; h+=stride_y){
			for (uint w = startWidth + idx; w < endWidth; w+=stride_x){
				float4 sum = float4(0, 0, 0, 0);
				float8 readImage8;
				for (uint i = 0; i < heightKernel; i++){
					for (uint j = 0; j < widthKernel; j++){
						readKernel = convert_float(ptrKernel[ i * stepsWidthKernel + j ]);
						const int index = j % 5;
						if (index == 0){
							int indexImageY = h - partHeightKernel + i;
							int indexImageX = w - partWidthKernel + j;
							readImage8.s0123 = convert_float4(ptrImage[  indexImageY * stepsWidth + indexImageX ] ); 
							sum += readImage8.s0123 * readKernel;
						}else if (index == 1){
							int indexImageY = h - partHeightKernel + i;
							int indexImageX = w - partWidthKernel + j;
							readImage8.s4567 = convert_float4(ptrImage[  indexImageY * stepsWidthImage + indexImageX ] ); 
							sum += readImage8.s1234 * readKernel;
						}else if (index == 2){
							sum += readImage8.s2345 * readKernel;
						}else if (index == 3){
							sum += readImage8.s3456 * readKernel;
						}else if (index == 4){
							sum += readImage8.s4567 * readKernel;
						}
					}
				}
				ptrImage[  indexImageY * stepsWidthImage + indexImageX ] = convert_uchar4(sum / operations);
			}
		}
							
		for (uint h = idy; h < partHeightKernel; h+=stride_y){
			for (uint w = idx; w < partWidthKernel; w+stride_x){
				float sum = 0.0f;
				for (uint i = 0; i < heightKernel; i++){
					for (uint j = 0; j < widthKernel; j++){
						int indexImageY = h - partHeightKernel + i;
						int indexImageX = w - partWidthKernel + j;
						if (indexImageY >= 0 && indexImageX >= 0){
							readImage = convert_float(image[  indexImageY * _width + indexImageX ] ); 
							readKernel = convert_float(kernel[ i * stepsWidthKernel + j ]);
							sum += readImage * readKernel;
						}
					}
				}
				ptrImage[  indexImageY * stepsWidth + indexImageX ] = convert_uchar(sum / operations);
			}
		}		

		for (uint h = height - partHeightKernel + idy; h < height; h+=stride_y){
			for (uint w = stepsWidthImage - partWidthKernel - 4 + idx; w < width; w+=stride_x){
				float sum = 0.0f;
				for (uint i = 0; i < heightKernel; i++){
					for (uint j = 0; j < widthKernel; j++){
						int indexImageY = h - partHeightKernel + i;
						int indexImageX = w - partWidthKernel + j;
						if (indexImageY < height && indexImageX < width){
							readImage = convert_float(image[  indexImageY * _width + indexImageX ] ); 
							readKernel = convert_float(kernel[ i * stepsWidthKernel + j ]);
							sum += readImage * readKernel;
						}
					}
				}
				ptrImage[  indexImageY * stepsWidth + indexImageX ] = convert_uchar(sum / operations);
			}
		}		
}
)==="