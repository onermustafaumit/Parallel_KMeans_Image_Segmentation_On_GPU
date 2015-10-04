#include "StdAfx.h"
#include <opencv\highgui.h>
#include <opencv\cv.h>
#include "opencv2/core/core.hpp"
#include <iostream>
#include <windows.h>

using namespace cv;
using namespace std;

//#define threshold 3000 //square_dist threshold value
#define max_num_of_clusters 25
 
int main()
{
	Mat image;
	
	const char *filename;
	string file_name, segmented_filename;
	cout << "Please enter an image number: ";
	getline (cin, file_name);
	segmented_filename=file_name+"_Segmented"+"_CPU.jpg";
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
	Mat_<Vec3b>::iterator it = image.begin<Vec3b>(), itEnd = image.end<Vec3b>();
	unsigned char cluster_intensity[3*max_num_of_clusters]={0};
	float temp_cluster_intensity[3*max_num_of_clusters]={0};
	int num_of_cluster_members[max_num_of_clusters]={0};
	unsigned char blue, green, red;

	width=image.cols;
	height=image.rows;

	// Find number of clusters and initial cluster centroids
	int threshold;
	cout << "Enter an integer value as a threshold to be used in initialization: ";
	cin >> threshold;
	
	num_of_clusters=0;
	unsigned char* labels;
	labels = (unsigned char*) malloc (height*width*sizeof(unsigned char));
	unsigned char* segmented_image;
	segmented_image=(unsigned char*) malloc (3*height*width*sizeof(unsigned char));

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
				diff_blue=cluster_intensity[3*k]-blue;
				diff_green=cluster_intensity[3*k+1]-green;
				diff_red=cluster_intensity[3*k+2]-red;
				square_dist=diff_blue*diff_blue + diff_green*diff_green + diff_red*diff_red;
				if (square_dist < threshold)
				{
					cluster_intensity[3*k]=0.5*(cluster_intensity[3*k]+blue);
					cluster_intensity[3*k+1]=0.5*(cluster_intensity[3*k+1]+green);
					cluster_intensity[3*k+2]=0.5*(cluster_intensity[3*k+2]+red);
					break;
				}
				if (k==num_of_clusters)
				{
					if (num_of_clusters<max_num_of_clusters)
					{
						cluster_intensity[3*num_of_clusters]=blue;
						cluster_intensity[3*num_of_clusters+1]=green;
						cluster_intensity[3*num_of_clusters+2]=red;
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
	
	// K-means segmentation
	int num_of_iterations=10;
	float temp_sqr_dist=195075; // Max theoretical distance
	int temp_cluster=0;
	int k=0;
	
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	
	for (int l=0; l<num_of_iterations; l=l+1)
	{

		for (int i=0; i<height; i=i+1 )
		{
			for (int j=0; j<width; j=j+1 )
			{
				temp_cluster=0;
				temp_sqr_dist=195075; // Max theoretical distance

				intensity = image.at<Vec3b>(i, j);
				blue = intensity.val[0];
				green = intensity.val[1];
				red = intensity.val[2];

				for (k=0; k<num_of_clusters; k++)
				{
					diff_blue=cluster_intensity[3*k]-blue;
					diff_green=cluster_intensity[3*k+1]-green;
					diff_red=cluster_intensity[3*k+2]-red;
					square_dist=diff_blue*diff_blue + diff_green*diff_green + diff_red*diff_red;
					if(square_dist<temp_sqr_dist)
					{
						temp_sqr_dist=square_dist;
						temp_cluster=k;
					}
				}
				// label assignment
				labels[i*width+j]= temp_cluster;

				// Cluster centroid temp variables
				temp_cluster_intensity[3*temp_cluster]=temp_cluster_intensity[3*temp_cluster]+blue;
				temp_cluster_intensity[3*temp_cluster+1]=temp_cluster_intensity[3*temp_cluster+1]+green;
				temp_cluster_intensity[3*temp_cluster+2]=temp_cluster_intensity[3*temp_cluster+2]+red;
				num_of_cluster_members[temp_cluster]=num_of_cluster_members[temp_cluster]+1;
			}
		}

		// Update Cluster Centroids
		for (int m=0; m<num_of_clusters; m++)
		{
			cluster_intensity[3*m]=temp_cluster_intensity[3*m]/num_of_cluster_members[m];
			cluster_intensity[3*m+1]=temp_cluster_intensity[3*m+1]/num_of_cluster_members[m];
			cluster_intensity[3*m+2]=temp_cluster_intensity[3*m+2]/num_of_cluster_members[m];
		}

		memset(temp_cluster_intensity, 0, sizeof(temp_cluster_intensity));
		memset(num_of_cluster_members, 0, sizeof(num_of_cluster_members));

	}
	
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
			segmented_image[3*(i*width+j)]=cluster_intensity[3*labels[i*width+j]];
			segmented_image[3*(i*width+j)+1]=cluster_intensity[3*labels[i*width+j]+1];
			segmented_image[3*(i*width+j)+2]=cluster_intensity[3*labels[i*width+j]+2];
		}

	}

	Mat src =  Mat(height,width, CV_8UC3, segmented_image);
	namedWindow(segmented_filename,CV_WINDOW_NORMAL); 
	imwrite(segmented_filename,src);
    imshow(segmented_filename, src);
	waitKey(0);
	
	free (labels);

	system("PAUSE");
    return 0; 
}