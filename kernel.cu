
#include <stdio.h>
#include "wb.h"

using namespace std;

#define THREADS_PER_BLOCK 1024 //max number threads per block
#define SECTION_SIZE 256 //for color size 

struct Image
{
	int width;
	int height;
	int channels;
	int colors;
	float* data;
//Image(int imageWidth = 0, int imageHeight = 0, int imageChannels = wbInternal::kImageChannels, int imageColors = wbInternal::kImageColorLimit) : width(imageWidth), height(imageHeight), channels(imageChannels), colors(imageColors), data(NULL)
	 Image(int imageWidth = 0, int imageHeight = 0, int imageChannels = 1, int imageColors = wbInternal::kImageColorLimit) : width(imageWidth), height(imageHeight), channels(imageChannels), colors(imageColors), data(NULL) // image channel is 1
	 // image channel is 1 for gray image and image color is 255.
	{
		const int numElements = width * height * channels;
		// Prevent zero-length memory allocation
		if (numElements > 0)
			data = new float[numElements];
	}
};


Image wb_import(const char* fName)
{
	ifstream inFile(fName, ios::binary);
	
	if (!inFile.is_open())
	{
		cerr << "Error opening image file " << fName << ". " << wbInternal::wbStrerror(errno) <<  endl;
		exit(EXIT_FAILURE);		
	}
	// Read PGM image header
	string magic;
	getline(inFile, magic);
	
	// use P2 format	
	if (magic != "P2") 
	{
		cerr << "Error reading image file " << fName << ". " << "Expecting 'P2' image format but got '" << magic << "'" << endl;
		inFile.close();
		exit(EXIT_FAILURE);
	}
	
	// Filter image comments
	if (inFile.peek() == '#')
	{
		string commentStr;
		getline(inFile, commentStr);
	}

	Image image;	
	inFile >> image.width;
	
	if (inFile.fail() || 0 >= image.width)
	{
		 cerr << "Error reading width of image in file " << fName <<  endl;
		 inFile.close();
		 exit(EXIT_FAILURE);
	}
	
	inFile >> image.height;
	
	if (inFile.fail() || 0 >= image.height)
	{
		 cerr << "Error reading height of image in file " << fName <<  endl;
		 inFile.close();
		 exit(EXIT_FAILURE);
	}

	inFile >> image.colors;
	if (inFile.fail() || image.colors > wbInternal::kImageColorLimit)
	{
		 cerr << "Error reading colors value of image in file " << fName <<  endl;
		inFile.close();
		 exit(EXIT_FAILURE);
	}
	
	while (isspace(inFile.peek()))
    {
        inFile.get();
    }
	 // not need raw data 
	const int numElements = image.width * image.height * image.channels;

	float* data = new float[numElements];
	
	for (int i = 0; i < numElements; i++)
	{
		inFile >> data[i];		
	}
	
	inFile.close();	
	image.data = data;
	return image;
}

void  wb_save(const Image& image, const char* fName) {
	 ostringstream oss;
		
	//oss << "P6\n" << "# Created by applying convolution " << wbArg_getInputFile(args, args.argc - 3) << "\n" << image.width << " " << image.height << "\n" << image.colors << "\n";
	  oss << "P2\n" << "#  Created by applying histogram "  << "\n" <<  image.width << " " << image.height << "\n" << image.colors << "\n";
	 string headerStr(oss.str());

	 ofstream outFile(fName,  ios::binary);
	 outFile.write(headerStr.c_str(), headerStr.size());

	 const int numElements = image.width * image.height * image.channels;

	for (int i = 0; i < numElements; ++i)
	{
		outFile << (int)image.data[i] << " ";
		//printf("image data %d \n" ,  image.data[i]);
	}

	outFile.close();
}

int wbImage_getWidth(const Image& image)
{
	return image.width;
}

int wbImage_getHeight(const Image& image)
{
	return image.height;
}

int wbImage_getChannels(const Image& image)
{
	return image.channels;
}

float* wbImage_getData(const Image& image)
{
	return image.data;
}

Image awbImage_new(const int imageWidth, const int imageHeight, const int imageChannels)
{
	Image image(imageWidth, imageHeight, imageChannels);
	return image;
}

void wbImage_delete(Image& image)
{
	delete[] image.data;
}

__global__ void histogram(float *buffer, int size, float *histo) 
{	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// stride is total number of threads
	int stride = blockDim.x * gridDim.x;
	//  All threads handle blockDim.x * gridDim.x consecutive elements
	while (i < size) 
	{	
		atomicAdd(&(histo[(int)buffer[i]]), 1);
		i += stride;
	}
}

__global__ void private_histogram(float *histo, float *buffer, int size)
{	
	__shared__ unsigned int private_histogram[SECTION_SIZE];
	if (threadIdx.x < SECTION_SIZE)
	{	
		private_histogram[threadIdx.x] = 0;
	}
	__syncthreads();
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	while (i < size)
	{	
		atomicAdd(&(private_histogram[(int)buffer[i]]), 1);
		i += stride;	
	}
	__syncthreads();

	if (threadIdx.x < SECTION_SIZE)
	{
		atomicAdd(&(histo[threadIdx.x]), private_histogram[threadIdx.x]);
	}
}

// Kogge-Stone code for scan
__global__ void kogge_stone(float *histo, float *scanning, int size, float *swap) 
{	
	__shared__ float scan[SECTION_SIZE]; //assume it is equal to block size
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{	
		scan[threadIdx.x] = histo[i];
	}
	// the code below performs iterative scan 

	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
	{
		__syncthreads();
		if (threadIdx.x >= stride)
		{	
			scan[threadIdx.x] += scan[threadIdx.x - stride];
		}
	}
			
	__syncthreads();
	  for(int i = 0 ; i< SECTION_SIZE ; i++ )
	 {
		 swap[i] = scan[i];		 
	 }	
}

// Brent-Kung code for  scan
__global__ void brent_kung(float *histo, float *scanning, int size ,float *swap )
{	
	__shared__ float scan[2 * SECTION_SIZE];
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < SECTION_SIZE)
	{	
		scan[threadIdx.x] = histo[i];
	}
		
	for (unsigned int stride = 1; stride <= blockDim.x ; stride *= 2)  
	{	
		__syncthreads();
		int index = (threadIdx.x + 1) * 2 * stride  - 1;
		
		if (index < blockDim.x)  
		{	
			scan[index] += scan[index - stride];
		}
		
	}
	for (unsigned int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) 
	{	
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < SECTION_SIZE) 
		{	
			scan[index + stride] += scan[index];
		}
	}
		
	__syncthreads();
	  for(int i = 0 ; i< SECTION_SIZE ; i++ )
	 {
		 swap[i] = scan[i];
		 		 
	 }
}
		
// find cdf min 
__global__ void find_cdx(float *swap , float *scanning , int size )
{		
	int cdf_min = swap[0];
	 // printf (" cdf_ min is  %d *********** \n " , cdf_min );
	
	for(int i = 1; i < SECTION_SIZE; i++ )
	{ 	
		if ((int)swap[i] < cdf_min )
		{
			cdf_min = swap[i];
			//printf (" cdf_ min is  %d *********** \n " , cdf_min );
		}		
	}
			
	scanning[threadIdx.x] = ((swap[threadIdx.x] - cdf_min) / (size - cdf_min) * (SECTION_SIZE - 1));				
}

// histogram equalization
__global__ void histo_equalize(float *output, float *input, int size, float *scanning)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < size)
	{
		output[i] = scanning[(int)input[i]];
	}
	__syncthreads();
}

int main(int argc, char ** argv) {	

 	int nDevices;
	cudaGetDeviceCount(&nDevices);
	printf("cudaGetDeviceCount: %d\n", nDevices);
	printf("There are %d CUDA devices.\n", nDevices);

	for (int i = 0; i < nDevices; i++) 
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d \n", i);
		printf("Device name: %s \n ", prop.name);
		printf("Block dimensions: %d x %d  x %d \n", prop.maxThreadsDim[0],prop.maxThreadsDim[1],  prop.maxThreadsDim[2]);
		printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);		
		printf ("Grid dimensions:  %d x %d x %d \n", prop.maxGridSize[0],  prop.maxGridSize[1],  prop.maxGridSize[2]);
		
	}
		
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	char * inputImageFile;
	char * outputImageFile;
	Image inputImage;
	Image outputImage;
	
	float * hostInputImageData;
	float * hostOutputImageData;
	float * hostHistogram; 
	
	inputImageFile = argv[1];
	outputImageFile = argv[2];	
	
	inputImage = wb_import(inputImageFile);
	hostHistogram = NULL;
	hostInputImageData = wbImage_getData(inputImage);	

	int imageWidth = wbImage_getWidth(inputImage);
	int imageHeight = wbImage_getHeight(inputImage);
	int imageChannels = wbImage_getChannels(inputImage);
    int imageDataSize = imageWidth * imageHeight * imageChannels; 

	 printf("%d %d %d\n", imageWidth, imageHeight, imageChannels);
	 printf("%f %f %f\n", hostInputImageData[0], hostInputImageData[1], hostInputImageData[2]);
	 printf ("image data size %d for determine grid size  \n " ,imageDataSize );
	
	outputImage = awbImage_new(imageWidth, imageHeight, imageChannels);
	hostOutputImageData = wbImage_getData(outputImage);

	float *deviceInputImageData; 
	float *deviceOutputImageData;
	float *deviceHistogram;
	float *scanning;
	float *swap;
	
  size_t size_image = imageWidth * imageHeight * imageChannels * sizeof(float);	 
	size_t size = SECTION_SIZE * sizeof(float);

	cudaMalloc((void **)&deviceInputImageData,size_image );
	cudaMalloc((void **)&deviceOutputImageData, size_image);
	cudaMalloc((void **)&deviceHistogram, size);
	cudaMalloc((void **)&scanning,size );
	cudaMalloc((void **)&swap,size );

	cudaMemcpy(deviceInputImageData, hostInputImageData, size_image, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceHistogram, hostHistogram,size, cudaMemcpyHostToDevice);	
	int grid_size = (imageDataSize + 1) / THREADS_PER_BLOCK;
	
    cudaEventRecord(start);
		
		//histogram <<< grid_size, THREADS_PER_BLOCK >>> (deviceInputImageData, imageDataSize, deviceHistogram);		
		private_histogram <<< grid_size, THREADS_PER_BLOCK >>> (deviceHistogram, deviceInputImageData, imageDataSize); // more efficiency
		
		//kogge_stone <<< 1, SECTION_SIZE >>> (deviceHistogram, scanning, imageDataSize, swap);
		brent_kung <<< 1, SECTION_SIZE >>> (deviceHistogram, scanning, imageDataSize, swap); //  more efficiency
		
		
		find_cdx<<< 1, SECTION_SIZE  >>> (swap , scanning , imageDataSize);
		histo_equalize <<< grid_size, THREADS_PER_BLOCK >>> (deviceOutputImageData, deviceInputImageData, imageDataSize, scanning);

    cudaEventRecord(stop);

	cudaMemcpy(hostOutputImageData, deviceOutputImageData, size_image, cudaMemcpyDeviceToHost);
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf(" run time is  %f  milliseconds \n " , milliseconds);

	wb_save(outputImage, outputImageFile);
	 
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceHistogram);
	cudaFree(scanning);

	return 0;
}
