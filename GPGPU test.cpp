// GPGPU test.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include "clDevice.h"

#ifdef _WIN32
#include <windows.h>
#include <gdiplus.h>
#include <Commdlg.h>
#include <immintrin.h>
#pragma comment(lib, "gdiplus.lib")
#endif
const char convolution[] =
#include "convolution.cl"
;
struct RGB {
	UINT8 b;
	UINT8 g;
	UINT8 r;
};
struct RGBA {
	UINT8 b;
	UINT8 g;
	UINT8 r;
	UINT8 a;
};
struct dataImage {
	size_t width;
	size_t height;
	size_t stride;
	int pixelFormat;
	RGBA* data;
};

enum FORMATS { BMP, JPG, GIF, TIF, PNG };

#define _BMP L"{557cf400-1a04-11d3-9a73-0000f81ef32e}"
#define _JPG L"{557cf401-1a04-11d3-9a73-0000f81ef32e}"
#define _GIF L"{557cf402-1a04-11d3-9a73-0000f81ef32e}"
#define _TIF L"{557cf405-1a04-11d3-9a73-0000f81ef32e}"
#define _PNG L"{557cf406-1a04-11d3-9a73-0000f81ef32e}"

dataImage WIN_load_image(const WCHAR* name) {
	dataImage image;
	Gdiplus::GdiplusStartupInput input;
	Gdiplus::GdiplusStartupOutput output;
	ULONG_PTR token;
	Gdiplus::GdiplusStartup(&token, &input, &output);
	Gdiplus::Color color;
	Gdiplus::Bitmap* bitmap = new Gdiplus::Bitmap(name);

	image.height = bitmap->GetHeight();
	image.width = bitmap->GetWidth();
	Gdiplus::BitmapData bitmapData = {};
	Gdiplus::Rect rect(0, 0, image.width, image.height);
	bitmap->LockBits(&rect, Gdiplus::ImageLockModeRead | Gdiplus::ImageLockModeWrite, PixelFormat32bppRGB, &bitmapData);

	RGBA*  pixels = (RGBA*)bitmapData.Scan0;
	UINT stride = abs(bitmapData.Stride) / 4;
	image.stride = bitmapData.Stride;
	image.pixelFormat = PixelFormat32bppRGB;
	image.data = (RGBA*)_mm_malloc(image.stride * image.height, 32);
	for (UINT row = 0; row < image.height; ++row)
		for (UINT col = 0; col < image.width; ++col)
		{
			image.data[row *	stride + col] = pixels[row * stride + col];
		}
	bitmap->UnlockBits(&bitmapData);
	delete bitmap;
	Gdiplus::GdiplusShutdown(token);

	return image;
}

void WIN_save_image(dataImage& image, const WCHAR* name, size_t format) {
	int cpu_info[4];
	int functions = 1;
	int sub_leaf = 0;
	__cpuidex(cpu_info, functions, sub_leaf);
	bool support_avx = cpu_info[2] & (1 << 28) || false;

	Gdiplus::GdiplusStartupInput input;
	Gdiplus::GdiplusStartupOutput output;
	ULONG_PTR token;
	Gdiplus::GdiplusStartup(&token, &input, &output);
	Gdiplus::Color color;
	Gdiplus::Bitmap* bitmap = new Gdiplus::Bitmap(image.width, image.height, image.stride, image.pixelFormat, (BYTE*)image.data);
	size_t length;

	CLSID сlsid;
	const WCHAR* _name = name;
	for (length = 0; _name[length] != 0; length++)
		length++;
	WCHAR * name_file = (WCHAR*)alloca((length + 6) * sizeof(WCHAR));
	for (size_t i = 0; i < length; i++)
		name_file[i] = name[i];

	switch (format) {
	case BMP:
		name_file[length] = L'.';
		name_file[length + 1] = L'b';
		name_file[length + 2] = L'm';
		name_file[length + 3] = L'p';
		name_file[length + 4] = 0;
		CLSIDFromString(_BMP, &сlsid);
		bitmap->Save(name_file, &сlsid, NULL);
		break;
	case JPG:
		name_file[length] = L'.';
		name_file[length + 1] = L'j';
		name_file[length + 2] = L'p';
		name_file[length + 3] = L'g';
		name_file[length + 4] = 0;
		CLSIDFromString(_JPG, &сlsid);
		bitmap->Save(name_file, &сlsid, NULL);
		break;
	case PNG:
		name_file[length] = L'.';
		name_file[length + 1] = L'p';
		name_file[length + 2] = L'n';
		name_file[length + 3] = L'g';
		name_file[length + 4] = 0;
		CLSIDFromString(_PNG, &сlsid);
		bitmap->Save(name_file, &сlsid, NULL);
		break;
	case TIF:
		name_file[length] = L'.';
		name_file[length + 1] = L't';
		name_file[length + 2] = L'i';
		name_file[length + 3] = L'f';
		name_file[length + 4] = L'f';
		name_file[length + 5] = 0;
		CLSIDFromString(_TIF, &сlsid);
		bitmap->Save(name_file, &сlsid, NULL);
		break;
	case GIF:
		name_file[length] = L'.';
		name_file[length + 1] = L'g';
		name_file[length + 2] = L'i';
		name_file[length + 3] = L'f';
		name_file[length + 4] = 0;
		CLSIDFromString(_GIF, &сlsid);
		bitmap->Save(name_file, &сlsid, NULL);
		break;
	}
	delete bitmap;
	Gdiplus::GdiplusShutdown(token);
}

int main()
{
//	float noise = (1.0f / (1.0f * sqrt(2.0f * 3.1415926535))) * exp(-pow(0.08 - 0.0f, 2) / 2.0f * 1.0f* 1.0f);
	cl_char kernel[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	clPlatform platform;
	cl_char* place = (cl_char*)_aligned_malloc(platform.getNumberDevices() * sizeof(clDevice), alignof(clDevice));
	clDevice** devices = (clDevice**)malloc(platform.getNumberDevices() * sizeof(clDevice*));
	cl_char* aligned_place = place;
	for (size_t i = 0; i < platform.getNumberDevices(); i++) {
		devices[i] = new(aligned_place) clDevice(&platform, i);
		aligned_place += sizeof(clDevice);
	}
	dataImage image = WIN_load_image(L"earth.png");
	size_t i = 0;
	if (i < platform.getNumberDevices()) {
		void* data[] = { (void*)image.data, (void*)kernel, NULL };
		size_t length_data[] = { image.height * image.stride, sizeof(kernel), image.height * image.stride };
		cl_uint indices_memory[] = { 0, 1, 2 };
		cl_uint index_kernel_buffer[] = { 0, 3, 6 };
		cl_uint index_kernel_arguments[] = { 1, 2, 4, 5 };
		size_t work_size[] = { image.stride / 4,image.height, 1 };
		cl_uint arguments[] = { image.stride / 4,image.height , 12 / 4, 3 };
		cl_uchar type_arguments[] = { sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint),  sizeof(cl_uint) };
		devices[i]->clPushProgram((cl_char*)convolution, sizeof(convolution), NULL);
		devices[i]->clPushKernel((cl_char*)"convolution_rgba", sizeof("convolution_rgba"));
		devices[i]->clPushKernel((cl_char*)"noise_rgba", sizeof("noise_rgba"));
		printf("\n%s \n", devices[i]->getNameKernel(0));
		devices[i]->mallocBufferMemory((const void**)data, length_data, 3, sizeof(char));
		devices[i]->setArguments(0, indices_memory, 3, index_kernel_buffer, arguments, type_arguments, 4, index_kernel_arguments);
		devices[i]->startCalculate(0, work_size);
		float m_exp = 0.0f;
		float s_div = 1.0f;
		arguments[2] = *(int*)&m_exp;
		arguments[3] = *(int*)&s_div;
		index_kernel_arguments[2] = 3;
		index_kernel_arguments[3] = 4;
		devices[i]->setArguments(1, &indices_memory[2], 1, index_kernel_buffer, arguments, type_arguments, 4, index_kernel_arguments);
		devices[i]->startCalculate(1, work_size);
		type_arguments[i] = sizeof(cl_uchar);
		devices[i]->readData((void**)&image.data, &indices_memory[2], &type_arguments[0], &length_data[0], 1);
	}
	WIN_save_image(image, L"result", PNG);

	for (size_t i = 0; i < platform.getNumberDevices(); i++) {
		devices[i]->~clDevice();
	}
	free(devices);
	_aligned_free(place);
}