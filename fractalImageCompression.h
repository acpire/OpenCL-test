#pragma once
#include "clDevice.h"
class fractalImageCompression
{
public:
	fractalImageCompression(clDevice* device, cl_uchar4* image, size_t width, size_t height);
	~fractalImageCompression();
};

