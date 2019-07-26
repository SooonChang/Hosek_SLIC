#ifndef SLIC2_H
#define SLIC2_H

/* slic.h.
*
* Written by: Pascal Mettes.
*
* This file contains the class elements of the class Slic. This class is an
* implementation of the SLIC Superpixel algorithm by Achanta et al. [PAMI'12,
* vol. 34, num. 11, pp. 2274-2282].
*
* This implementation is created for the specific purpose of creating
* over-segmentations in an OpenCV-based environment.
*/

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
#include <opencv2/opencv.hpp>
using namespace std;

/* 2d matrices are handled by 2d vectors. */
#define vec2dd vector<vector<double> >
#define vec2di vector<vector<int> >
#define vec2db vector<vector<bool> >
#define vec2dP vector<vector<cv::Point3d> >

#define vec3di vector<vector<vector<int>> >
/* The number of iterations run by the clustering algorithm. */
#define NR_ITERATIONS 10

class DataTerm
{
public:
	vec2dd matchingCost;
	vec2dd subtract_MatchingCost;
	vec2dd subtract_ColorCost;

	vec2di connectedNode;
	int connectedNodeTerm[10];
};

/*
* class Slic.
*
* In this class, an over-segmentation is created of an image, provided by the
* step-size (distance between initial cluster locations) and the colour
* distance parameter.
*/


class Slic : public DataTerm {
public:
	/* The cluster assignments and distance values for each pixel. */

	vec2di FOV_clusters; // Cluster의 index.
	vec2dd FOV_distances; // j번째 Cluster와 Pixel i와의 거리.

	/* The LAB and xy values of the centers. */

	vec2dd FOV_centers; // Cluster의 L,a,b,x,y값.

	/* The number of occurences of each center. */

	vector<int> FOV_center_counts;

	/* The step size per cluster, and the colour (nc) and distance (ns)
* parameters. */
	int step, nc, ns;

public:
	/* The BGR values to the centers of HW model and superpixel. */
	vec2dd HW_centers;
	vec2dd SP_centers;
	vec2dd FOV_HW_centers;
	vec2dd FOV_SP_centers;

	vec2dd HW_variance;
	vec2dd HW_tot_variance;
	vec2dd FOV_HW_variance;
	vec2dd FOV_HW_tot_variance;

	vec2dd selected_SP_centers;
	vec2dd unselected_SP_centers;

	vec2dd selected_HW_centers;
	vec2dd unselected_HW_centers;

	vec2dd gainMeanValue;

	vector<double> variance;
	vector<int> turbidity;
	vector<double> albedo;

	vector<double> mean_variance;




public:
	/* Compute the distance between a center and an individual pixel. */
	double compute_dist(int ci, CvPoint pixel, CvScalar colour);
	/* Find the pixel with the lowest gradient in a 3x3 surrounding. */
	CvPoint find_local_minimum(IplImage* image, CvPoint center);

	/* Remove and initialize the 2d vectors. */
	void clear_data();
	void init_data(IplImage* image);

public:
	/* Class constructors and deconstructors. */
	Slic();
	~Slic();

	/* Generate an over-segmentation for an image. */
	void generate_superpixels(IplImage* image, int step, int nc);
	/* Enforce connectivity for an image. */
	void create_connectivity(IplImage* image);

	/* Draw functions. Resp. displayal of the centers and the contours. */
	void display_center_grid(IplImage* image, CvScalar colour);
	void display_contours(IplImage* image, CvScalar colour);
	void colour_with_cluster_means(IplImage* image);

public:
	//cv::Mat lastSelectHWmodel;
	//void slic_superpixel_HWcenters_Initialize(cv::Mat HW_Image, cv::Mat SP_Image, int width, int height);
	//void meanHigher_iteration(cv::Mat fisheyeImage, cv::Mat HWmodel[], double meanHigherArray[], int width, int height);

};

#endif
