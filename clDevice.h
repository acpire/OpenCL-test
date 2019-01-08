

#include "CL/cl.h"
#include <malloc.h>
#include <stdio.h>
#include <corecrt_memcpy_s.h>
#include <new>
#pragma comment(lib, "x86_64/OpenCL.lib")
#pragma warning(disable:4996)


struct structDeviceInfo {
	cl_uint maxComputeUnit;
	cl_uint workItemDemension;
	cl_uint maxComputeUnits;
	cl_uint preferVectorChar;
	cl_uint preferVectorShort;
	cl_uint preferVectorInt;
	cl_uint preferVectorLong;
	cl_uint preferVectorFloat;
	cl_uint preferVectorDouble;
	cl_bool supportImages;
	cl_char* deviceVendor;
	cl_char* deviceExtensions;
	size_t workItemSizes[3];
	size_t maximumParametersInKernel;
	size_t maxWorkGroupSize;
	size_t maxHeightImage2D;
	size_t maxWidthImage2D;
	size_t maxHeightImage3D;
	size_t maxWidthImage3D;
	size_t maxDepthImage3D;
	cl_ulong localMemorySize;
	cl_ulong maxGlobalMemoryAllocate;
	cl_ulong globalMemSize;
	cl_device_type deviceType;
	cl_device_local_mem_type localMemoryType;
};


#define CL_CHECK(codeError, stringError)											\
   do {																				\
     if (codeError == CL_SUCCESS)													\
       break;																		\
     printf("\nOpenCL Error: '%s' returned %s!\n", stringError, getInformationError(codeError));	\
     abort();																		\
    } while (0)



class clPlatform
{
	cl_uint numberPlatforms;
	cl_platform_id* platforms;
	cl_uint* platformDevices;
	cl_uint numberDevices;
	cl_device_id* devices;
	cl_command_queue* queue;
	cl_context* context;
	cl_uint getNextPlatform;
public:
	clPlatform();
	~clPlatform();
	cl_platform_id* getPlatformID(cl_uint index);
	cl_device_id* getDeviceID(cl_uint index);
	cl_command_queue* getCommandQueueID(cl_uint index);
	cl_context* getContextID(cl_uint index);
	cl_uint getNumberDevices() { return numberDevices; };
};

class clDevice
{

	cl_char* profileVersionNameVendorExtensions[5];
	structDeviceInfo DeviceInfo;

	cl_platform_id* platform;
	cl_context* context;
	cl_device_id* device;
	cl_command_queue* queue;
	cl_mem* ptrMemoryDevice;

	cl_char** namesPrograms;
	cl_char** namesKernels;
	cl_program* programDevice;
	cl_kernel* kernels;
	size_t numberKernels;
	size_t numberPrograms;
	size_t numberObjectMemory;
public:
	clDevice(clPlatform* platformData, cl_uint indexDevice);
	bool clPushProgram(cl_char* text, size_t lengthText, const cl_char* options);
	bool clPushKernel(cl_char * text, size_t lengthText);
	cl_bool mallocBufferMemory(const void ** data, size_t * lengthData, size_t numberArrays, size_t lengthType);
	cl_bool setArguments(cl_uint indexKernel, cl_uint* indicesMemoryBuffer, cl_uint numberIndicesMemoryBuffer, cl_uint* index_kernel_buffer, void* arguments, cl_uchar* typeArgubents, cl_uint numberArguments, cl_uint* index_kernel_arguments);
	cl_bool startCalculate(cl_uint indexKernel, size_t globalWork[3]);
	cl_bool readData(void ** returnedData, cl_uint * indicesReadData, cl_uchar * typeArgubentsReturnedData, cl_ulong * lengthWrite, cl_uint numberIndicesReadData);
	cl_char* getNameKernel(cl_uint index);
	cl_char* getNameProgram(cl_uint index);
	~clDevice();
};
