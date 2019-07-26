/*
* test_slic.cpp.
*
* Written by: Pascal Mettes.
*
* This file creates an over-segmentation of a provided image based on the SLIC
* superpixel algorithm, as implemented in slic.h and slic.cpp.
*/

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
using namespace std;

#include "slic2.h"

int main(int argc, char* argv[]) {
	/* Load the image and convert to Lab colour space. */
	IplImage* image = cvLoadImage("Input_Image/subImage2.png", 1);
	IplImage* lab_image = cvCloneImage(image);
	cvCvtColor(image, lab_image, CV_BGR2Lab);

	/* Yield the number of superpixels and weight-factors from the user. */
	int w = image->width, h = image->height;
	int nr_superpixels = atoi("2000");
	int nc = atoi("200");

	double step = sqrt((w * h) / (double)nr_superpixels);

	/* Perform the SLIC superpixel algorithm. */
	Slic slic;
	slic.generate_superpixels(lab_image, step, nc);
	slic.create_connectivity(lab_image);
	slic.colour_with_cluster_means(image);

	/* Display the contours and show the result. */
	slic.display_contours(image, cv::Scalar(0,0,255));
	//slic.display_center_grid(image, CV_RGB(0, 255, 0));

	cvShowImage("result", image);
	cvWaitKey(0);

	return 0;
	
}