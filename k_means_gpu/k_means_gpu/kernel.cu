#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv\highgui.h>
#include <opencv\cv.h>
#include "opencv2/core/core.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <tchar.h>
#include <windows.h>

using namespace cv;
using namespace std;

//#define threshold 3000 //square_dist threshold value
#define max_num_of_clusters 25

__constant__ unsigned char d_cluster_intensity[3*25*sizeof(unsigned char)]; // Declare constant memory for cluster intensities

__global__ void cluster_assignment( unsigned char *d_image_ptr, unsigned char *d_labels, int num_of_clusters, int width, int height)
{
	//extern __shared__ unsigned char temp[];	//one shared temp array for each block to store pixel color values
	int col_num = blockDim.x * blockIdx.x + threadIdx.x;
	int row_num = blockDim.y * blockIdx.y + threadIdx.y;
	int index_d_image = 3*(row_num * width + col_num);
	//int temp_index=blockDim.x*threadIdx.y + threadIdx.x;
	
	unsigned char blue, green, red;
	float diff_blue, diff_green, diff_red, square_dist;
	float temp_sqr_dist=195075; // Max theoretical distance
	unsigned char temp_cluster;

	if(row_num < height && col_num < width)
	{
		//temp[temp_index] = d_image_ptr[index_d_image]; //Read b value from global memory and write to shared memory
		//temp[temp_index+blockDim.x *blockDim.y] = d_image_ptr[index_d_image+1]; //Read g value from global memory and write to shared memory
		//temp[temp_index+blockDim.x *blockDim.y*2] = d_image_ptr[index_d_image+2]; //Read r value from global memory and write to shared memory

		//__syncthreads();

		blue = d_image_ptr[index_d_image]; //Read b value from global memory and write to shared memory
		green = d_image_ptr[index_d_image+1]; //Read g value from global memory and write to shared memory
		red = d_image_ptr[index_d_image+2]; //Read r value from global memory and write to shared memory

		for (int k=0; k<num_of_clusters; k=k+1)
		{
			diff_blue=(float)blue-(float)d_cluster_intensity[3*k];
			diff_green=(float)green-(float)d_cluster_intensity[3*k+1];
			diff_red=(float)red-(float)d_cluster_intensity[3*k+2];
			square_dist=diff_blue*diff_blue + diff_green*diff_green + diff_red*diff_red;

			if(square_dist < temp_sqr_dist)
			{
				temp_sqr_dist=square_dist;
				//temp_cluster=static_cast<unsigned char>(k);
				temp_cluster=(unsigned char)k;
			}

		}

		d_labels[row_num * width + col_num]=temp_cluster;
	}
}


int main()
{
	Mat image;
	
	const char *filename;
	string file_name, segmented_filename;
	cout << "Please enter an image number: ";
	getline (cin, file_name);
	segmented_filename=file_name+"_Segmented"+"_GPU.jpg";
	file_name=file_name + ".jpg";
	filename=file_name.c_str();

    image = imread(filename, 1);
    if( image.empty() )
    {
        cout << "Couldn't open image " << filename << "\n";
        return 0;
    }
    namedWindow(file_name,CV_WINDOW_NORMAL);
    imshow(file_name, image);
	//waitKey(1);

	// Image properties
	int width, height, num_of_clusters;
	float diff_blue, diff_green, diff_red, square_dist; // channel value differences and square distance 
	
	Vec3b intensity;

	unsigned char h_cluster_intensity[3*max_num_of_clusters]={0};
	unsigned char blue, green, red;

	width=image.cols;
	height=image.rows;

	// Find number of clusters and initial cluster centroids by using a sampling procedure
	int threshold;
	cout << "Enter an integer value as a threshold to be used in initialization: ";
	cin >> threshold;

	num_of_clusters=0;

	for (int i=3; i<height; i=i+3)
	{
		for (int j=3; j<width; j=j+3)
		{
			intensity = image.at<Vec3b>(i, j);
			blue = intensity.val[0];
			green = intensity.val[1];
			red = intensity.val[2];
			for (int k=0; k<=num_of_clusters; k++)
			{
				diff_blue=h_cluster_intensity[3*k]-blue;
				diff_green=h_cluster_intensity[3*k+1]-green;
				diff_red=h_cluster_intensity[3*k+2]-red;
				square_dist=diff_blue*diff_blue + diff_green*diff_green + diff_red*diff_red;
				if (square_dist < threshold)
				{
					h_cluster_intensity[3*k]=0.5*(h_cluster_intensity[3*k]+blue);
					h_cluster_intensity[3*k+1]=0.5*(h_cluster_intensity[3*k+1]+green);
					h_cluster_intensity[3*k+2]=0.5*(h_cluster_intensity[3*k+2]+red);
					break;
				}
				if (k==num_of_clusters)
				{
					if (num_of_clusters<max_num_of_clusters)
					{
						h_cluster_intensity[3*num_of_clusters]=blue;
						h_cluster_intensity[3*num_of_clusters+1]=green;
						h_cluster_intensity[3*num_of_clusters+2]=red;
						num_of_clusters=num_of_clusters+1;
						break;
					}
					else
					{
						cout << "Max number of clusters are exceeded!!!" ;
						system("PAUSE");
						return 0;
					}
				}
			}
		}
	}
	cout << "Number of clusters:" << num_of_clusters << endl;
	

	//Kernel Parameters
	dim3 grid, block;
	block.x= 16 ;
	block.y= 16 ;
	grid.x= width/block.x; 
	grid.y= height/block.y;

	// Pointers
	unsigned char *h_labels, *d_labels;
	unsigned char *h_segmented_image;
	unsigned char *h_image_ptr, *d_image_ptr; // Host and Device image pointers
	float *d_temp_cluster_intensity, *h_temp_cluster_intensity;
	unsigned int *d_num_of_cluster_members, *h_num_of_cluster_members;
	
	// Host Memory Allocations
	h_labels = (unsigned char*) malloc (height*width*sizeof(unsigned char));
	h_segmented_image=(unsigned char*) malloc (3*height*width*sizeof(unsigned char));
	h_temp_cluster_intensity=(float*) malloc (3*num_of_clusters*sizeof(float));
	h_num_of_cluster_members=(unsigned int*) malloc (num_of_clusters*sizeof(unsigned int));
	h_image_ptr=image.ptr(0); // Host image pointer
	memset(h_temp_cluster_intensity, 0, 3*num_of_clusters*sizeof(float));
	memset(h_num_of_cluster_members, 0, num_of_clusters*sizeof(unsigned int));

	// Device Memory Allocations
	cudaMalloc((void**)&d_labels,height*width*sizeof(unsigned char));
	cudaMalloc((void**)&d_image_ptr,3*height*width*sizeof(unsigned char));

	// Device Memset
	cudaMemset(d_labels, (unsigned char)0, height*width*sizeof(unsigned char));

	// K-means segmentation
	int num_of_iterations=10;
	int temp_cluster;
	int index;

	//Create timer events
	/*cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float et; //elapsed time
	cudaEventRecord(start, 0);*/

	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);

	// Host to Device Initial Memcpy
	cudaMemcpy(d_image_ptr,h_image_ptr,3*height*width*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_cluster_intensity,h_cluster_intensity,3*max_num_of_clusters*sizeof(unsigned char)); // Copy host data to constant memory

	for (int l=0; l<num_of_iterations; l=l+1)
	{
		cluster_assignment<<<grid, block>>>(d_image_ptr, d_labels, num_of_clusters, width, height);
		cudaDeviceSynchronize();
		cudaMemcpy(h_labels, d_labels, height*width*sizeof(unsigned char), cudaMemcpyDeviceToHost);

		// Update Cluster Centroids
		for (int i=0; i<height; i=i+1 )
		{
			for (int j=0; j<width; j=j+1 )
			{
				index=i*width+j;
				temp_cluster=h_labels[index];

				h_temp_cluster_intensity[3*temp_cluster]=h_temp_cluster_intensity[3*temp_cluster]+h_image_ptr[3*(index)];
				h_temp_cluster_intensity[3*temp_cluster+1]=h_temp_cluster_intensity[3*temp_cluster+1]+h_image_ptr[3*(index)+1];
				h_temp_cluster_intensity[3*temp_cluster+2]=h_temp_cluster_intensity[3*temp_cluster+2]+h_image_ptr[3*(index)+2];
				
				h_num_of_cluster_members[temp_cluster]=h_num_of_cluster_members[temp_cluster]+1;
				
			}
		}

		for (int m=0; m<num_of_clusters; m++)
		{
			h_cluster_intensity[3*m]=h_temp_cluster_intensity[3*m]/h_num_of_cluster_members[m];
			h_cluster_intensity[3*m+1]=h_temp_cluster_intensity[3*m+1]/h_num_of_cluster_members[m];
			h_cluster_intensity[3*m+2]=h_temp_cluster_intensity[3*m+2]/h_num_of_cluster_members[m];
		}

		memset(h_temp_cluster_intensity, 0, sizeof(h_temp_cluster_intensity));
		memset(h_num_of_cluster_members, 0, sizeof(h_num_of_cluster_members));

		if (l<num_of_iterations-1)
		{
			cudaMemcpyToSymbol(d_cluster_intensity,h_cluster_intensity,3*max_num_of_clusters*sizeof(unsigned char)); // Copy host data to constant memory
		}

	}

	/*cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&et, start, stop);
	printf("Segmentation duration in ms: %f \n",et);
	std::cout << std::endl;*/

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

	//
	// We now have the elapsed number of ticks, along with the
	// number of ticks-per-second. We use these values
	// to convert to the number of elapsed microseconds.
	// To guard against loss-of-precision, we convert
	// to microseconds *before* dividing by ticks-per-second.
	//

	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	std::cout << "Elapsed Time in microseconds: " << ElapsedMicroseconds.QuadPart << std::endl;

	// Segmented Image Formation

	for (int i=0; i<height; i=i+1 )
	{
		for (int j=0; j<width; j=j+1 )
		{
			h_segmented_image[3*(i*width+j)]=h_cluster_intensity[3*h_labels[i*width+j]];
			h_segmented_image[3*(i*width+j)+1]=h_cluster_intensity[3*h_labels[i*width+j]+1];
			h_segmented_image[3*(i*width+j)+2]=h_cluster_intensity[3*h_labels[i*width+j]+2];
		}

	}

	Mat src =  Mat(height,width, CV_8UC3, h_segmented_image);
	namedWindow(segmented_filename,CV_WINDOW_NORMAL); 
	imwrite(segmented_filename,src);
	imshow(segmented_filename, src);
	waitKey(0);

	// Free Host Allocations
	free (h_labels);
	free (h_segmented_image);
	free (h_temp_cluster_intensity);
	free(h_num_of_cluster_members);

	// Free Device Allocations
	cudaFree(d_labels);
	cudaFree(d_image_ptr);

    system("PAUSE");
    return 0; 
}