// GPGPU test.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include "OpenCV.h"
#include "clDevice.h"
#include "InverseFilter.h"
#include "Wiener_filter.h"
#include "MakeNoise.h"
#include "fractalImageCompression.h"
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
const char fourier[] =
#include "fourier_transform.cl"
;
const char filters[] =
#include "filters.cl"
;
const char fractals[] =
#include "fractal.cl"
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
	dataImage image = WIN_load_image(L"earth.jpg");
	//cl_int2 coord_xy;
	//float rad = (3.1415926535f / 180.0f) * 180.0f;
	//cl_float2 center_rotate_xy;
	//size_t height = image.height;
	//size_t width = image.width;
	//for (int h = 0; h < height; h += 1) {
	//	for (int w = 0; w < width; w += 1) {
	//		if (w < width / 2 && h < height / 2) {
	//			center_rotate_xy.x = width / 4;
	//			center_rotate_xy.y = height / 4;
	//		}
	//		else if (w > width / 2 && h < height / 2) {
	//			center_rotate_xy.x = 3 * width / 4;
	//			center_rotate_xy.y = height / 4;
	//		}
	//		else if (w < width / 2 && h > height / 2) {
	//			center_rotate_xy.x = width / 4;
	//			center_rotate_xy.y = 3 * height / 4;
	//		}
	//		else if (w > width / 2 && h > height / 2) {
	//			center_rotate_xy.x = 3 * width / 4;
	//			center_rotate_xy.y = 3 * height / 4;
	//		}
	//		coord_xy.x = center_rotate_xy.x + (w - center_rotate_xy.x) * cos(rad) - (h - center_rotate_xy.y) * sin(rad);
	//		coord_xy.y = center_rotate_xy.y + (w - center_rotate_xy.x) * sin(rad) + (h - center_rotate_xy.y) * cos(rad);
	//		image.data[coord_xy.y * width + coord_xy.x] = image.data[h * width + w];


	//	}
	//}

	clPlatform platform;
	cl_char* place = (cl_char*)_aligned_malloc(platform.getNumberDevices() * sizeof(clDevice), alignof(clDevice));
	clDevice** devices = (clDevice**)malloc(platform.getNumberDevices() * sizeof(clDevice*));
	cl_char* aligned_place = place;
	for (size_t i = 0; i < platform.getNumberDevices(); i++) {
		devices[i] = new(aligned_place) clDevice(&platform, i);
		aligned_place += sizeof(clDevice);
		devices[i]->clPushProgram((cl_char*)convolution, sizeof(convolution), (cl_char*)"-cl-unsafe-math-optimizations -cl-fast-relaxed-math");
		devices[i]->clPushProgram((cl_char*)fourier, sizeof(fourier), (cl_char*)"-cl-unsafe-math-optimizations -cl-fast-relaxed-math");
		devices[i]->clPushProgram((cl_char*)filters, sizeof(filters), NULL);
	//	devices[i]->clPushProgram((cl_char*)fractals, sizeof(filters), NULL);

	}

	size_t i = 0;
	if (i < platform.getNumberDevices()) {
	//	MakeNoise noise(devices[i], (cl_uchar4*)image.data, image.width, image.height, (size_t)20, (size_t)20);
		//OpenCV cv_calc((char*)image.data, image.width, image.height);
		//Wiener_filter filter(devices[i], (cl_uchar4*)image.data, image.width, image.height, noise.getKernel(), (size_t)20, (size_t)20);
		//InverseFilter invFilter(devices[i], (cl_uchar4*)image.data, image.width, image.height, noise.getKernel(), (size_t)20, (size_t)2);
		fractalImageCompression fractal(devices[i], (cl_uchar4*)image.data, image.width, image.height, 16, 4);
	}
	WIN_save_image(image, L"result", PNG);

	//for (size_t i = 0; i < platform.getNumberDevices(); i++) {
	//	devices[i]->~clDevice();
	//}
	free(devices);
	_aligned_free(place);
}