#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"


#include <math.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <map>

#define DEFAULT_M 20
#define USE_DEFAULT_S -1
#define NR_ITERATIONS 10

using namespace std;

class SLIC
{
public:
	cv::Mat img; // original img
	cv::Mat img_f; // scaled to [0,1]
	cv::Mat img_lab; // converted to LAB colorspace
	cv::Mat show;

	cv::Mat labels; // Cluster의 index
	cv::Mat distances; // j번째 Cluster와 Pixel i와의 거리.
	
	vector<cv::Point> centers; // superpixel centers.
	vector<vector<float>> centers_value;
	vector<int> center_counts;

	float step, M; // step, compactness parameter
	int w,h; // cols and rows

	void init_data();
	float dist(int ci, cv::Point pixel, cv::Vec3f colour);
	cv::Point find_local_minimum(cv::Point c);
	void generateSuperpixels();

public:
	SLIC(cv::Mat& img, float M = DEFAULT_M, float k = USE_DEFAULT_S);
	cv::Mat display_contours(cv::Vec3b colour);
	cv::Mat colour_with_cluster_means();


	vector<cv::Point> getCenters();
	cv::Mat getLabels();

	

};