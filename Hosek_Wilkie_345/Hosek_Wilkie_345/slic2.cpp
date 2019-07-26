#include "slic2.h"
#include "ArHosekSkyModel.h"
#include "opencv2/opencv.hpp"
/*
* Constructor. Nothing is done here.
*/
Slic::Slic() {

}

/*
* Destructor. Clear any present data.
*/
Slic::~Slic() {
	clear_data();
}

/*
* Clear the data as saved by the algorithm.
*
* Input : -
* Output: -
*/
void Slic::clear_data() {
	FOV_clusters.clear();
	FOV_distances.clear();
	FOV_centers.clear();
	FOV_center_counts.clear();
	HW_centers.clear();
}


/*
* Initialize the cluster centers and initial values of the pixel-wise cluster
* assignment and distance values.
*
* Input : The image (IplImage*).
* Output: -
*/
void Slic::init_data(IplImage* image) {

	/* Initialize the cluster and distance matrices. */
	for (int i = 0; i < image->width; i++) {
		vector<int> FOV_cr;
		vector<double> FOV_dr;
		for (int j = 0; j < image->height; j++) {
			FOV_cr.push_back(-1);
			FOV_dr.push_back(FLT_MAX);
		}
		FOV_clusters.push_back(FOV_cr);
		FOV_distances.push_back(FOV_dr);
	}

	/* Initialize the centers and counters for FOV. */
	for (int i = step; i < image->width - step / 2; i += step) {
		for (int j = step; j < image->height - step / 2; j += step) {
			vector<double> FOV_center;
			/* Find the local minimum (gradient-wise). */
			CvPoint nc = find_local_minimum(image, cvPoint(i, j));
			CvScalar colour = cvGet2D(image, nc.y, nc.x);

			// ROI 원
			int x = (i - (image->width / 2)) * (i - (image->width / 2));
			int y = (j - (image->height / 2)) * (j - (image->height / 2));
			int r = (image->width / 2) * (image->width / 2);
			if (x + y <= r)
			{

				/* Generate the center vector. */
				FOV_center.push_back(colour.val[0]);
				FOV_center.push_back(colour.val[1]);
				FOV_center.push_back(colour.val[2]);
				FOV_center.push_back(nc.x);
				FOV_center.push_back(nc.y);

				/* Append to vector of centers. */
				FOV_centers.push_back(FOV_center);
				FOV_center_counts.push_back(0);
			}
		}
	}
}

/*
* Compute the distance between a cluster center and an individual pixel.
*
* Input : The cluster index (int), the pixel (CvPoint), and the Lab values of
*         the pixel (CvScalar).
* Output: The distance (double).
*/
double Slic::compute_dist(int ci, CvPoint pixel, CvScalar colour) {

	double dc = sqrt(pow(FOV_centers[ci][0] - colour.val[0], 2) + pow(FOV_centers[ci][1]
		- colour.val[1], 2) + pow(FOV_centers[ci][2] - colour.val[2], 2));
	double ds = sqrt(pow(FOV_centers[ci][3] - pixel.x, 2) + pow(FOV_centers[ci][4] - pixel.y, 2));

	return sqrt(pow(dc / nc, 2) + pow(ds / ns, 2));

	//double w = 1.0 / (pow(ns / nc, 2));
	//return sqrt(dc) + sqrt(ds * w);
}

/*
* Find a local gradient minimum of a pixel in a 3x3 neighbourhood. This
* method is called upon initialization of the cluster centers.
*
* Input : The image (IplImage*) and the pixel center (CvPoint).
* Output: The local gradient minimum (CvPoint).
*/

CvPoint Slic::find_local_minimum(IplImage* image, CvPoint center) {
	double min_grad = FLT_MAX;
	CvPoint loc_min = cvPoint(center.x, center.y);

	for (int i = center.x - 1; i < center.x + 2; i++) {
		for (int j = center.y - 1; j < center.y + 2; j++) {
			CvScalar c1 = cvGet2D(image, j + 1, i);
			CvScalar c2 = cvGet2D(image, j, i + 1);
			CvScalar c3 = cvGet2D(image, j, i);
			/* Convert colour values to grayscale values. */
			double i1 = c1.val[0];
			double i2 = c2.val[0];
			double i3 = c3.val[0];
			/*double i1 = c1.val[0] * 0.11 + c1.val[1] * 0.59 + c1.val[2] * 0.3;
			double i2 = c2.val[0] * 0.11 + c2.val[1] * 0.59 + c2.val[2] * 0.3;
			double i3 = c3.val[0] * 0.11 + c3.val[1] * 0.59 + c3.val[2] * 0.3;*/

			/* Compute horizontal and vertical gradients and keep track of the
			minimum. */
			if (sqrt(pow(i1 - i3, 2)) + sqrt(pow(i2 - i3, 2)) < min_grad) {
				min_grad = fabs(i1 - i3) + fabs(i2 - i3);
				loc_min.x = i;
				loc_min.y = j;
			}
		}
	}

	return loc_min;
}


/*
* Compute the over-segmentation based on the step-size and relative weighting
* of the pixel and colour values.
*
* Input : The Lab image (IplImage*), the stepsize (int), and the weight (int).
* Output: -
*/
void Slic::generate_superpixels(IplImage* image, int step, int nc) {
	this->step = step;
	this->nc = nc;
	this->ns = step;

	/* Clear previous data (if any), and re-initialize it. */
	clear_data();
	init_data(image);

	/* Run EM for 10 iterations (as prescribed by the algorithm). */
	for (int i = 0; i < NR_ITERATIONS; i++) {
		/* Reset distance values. */
		for (int j = 0; j < image->width; j++) {
			for (int k = 0; k < image->height; k++) {
				FOV_distances[j][k] = FLT_MAX;
			}
		}

		for (int j = 0; j < (int)FOV_centers.size(); j++) {
			/* Only compare to pixels in a 2 x step by 2 x step region. */
			for (int k = FOV_centers[j][3] - step; k < FOV_centers[j][3] + step; k++) {// 2S사이 x
				for (int l = FOV_centers[j][4] - step; l < FOV_centers[j][4] + step; l++) {// 2S사이 y

					if (k >= 0 && k < image->width && l >= 0 && l < image->height) {
						CvScalar colour = cvGet2D(image, l, k);
						double d = compute_dist(j, cvPoint(k, l), colour);

						/* Update cluster allocation if the cluster minimizes the
						distance. */
						if (d < FOV_distances[k][l]) {
							FOV_distances[k][l] = d;
							FOV_clusters[k][l] = j;
						}
					}
				}
			}
		}

		/* Clear the center values. */

		vector<vector<cv::Point2f>> pt;
		for (int j = 0; j < (int)FOV_centers.size(); j++) {
			FOV_centers[j][0] = FOV_centers[j][1] = FOV_centers[j][2] = FOV_centers[j][3] = FOV_centers[j][4] = 0;
			FOV_center_counts[j] = 0;

			pt.push_back(vector<cv::Point2f>());
		}

		/* Compute the new cluster centers. */
		for (int j = 0; j < image->width; j++) {
			for (int k = 0; k < image->height; k++) {
				int c_id = FOV_clusters[j][k];

				int x = (j - (image->width / 2)) * (j - (image->width / 2));
				int y = (k - (image->height / 2)) * (k - (image->height / 2));
				int r = (image->width / 2) * (image->width / 2);
				if (x + y <= r)
				{
					if (c_id != -1) {
						CvScalar colour = cvGet2D(image, k, j);

						FOV_centers[c_id][0] += colour.val[0];
						FOV_centers[c_id][1] += colour.val[1];
						FOV_centers[c_id][2] += colour.val[2];
						FOV_centers[c_id][3] += j;
						FOV_centers[c_id][4] += k;

						FOV_center_counts[c_id] += 1;


						cv::Point2f tmpPt = cv::Point2f(j, k);
						pt[c_id].push_back(tmpPt);
					}
				}
			}
		}

		/* Normalize the clusters. */
		for (int j = 0; j < (int)FOV_centers.size(); j++) {
			FOV_centers[j][0] /= FOV_center_counts[j];
			FOV_centers[j][1] /= FOV_center_counts[j];
			FOV_centers[j][2] /= FOV_center_counts[j];
			FOV_centers[j][3] /= FOV_center_counts[j];
			FOV_centers[j][4] /= FOV_center_counts[j];
		}


		///////////////////// Connected NodeTerm  /////////////////////////////
		for (int n = 0; n < FOV_centers.size(); ++n)
		{
			int minX = 9999, minY = 8888, maxX = -1, maxY = -1;
			for (int idx = 0; idx < pt[n].size(); ++idx)
			{
				if (minX > pt[n][idx].x) minX = pt[n][idx].x;
				if (minY > pt[n][idx].y) minY = pt[n][idx].y;
				if (maxX < pt[n][idx].x) maxX = pt[n][idx].x;
				if (maxY < pt[n][idx].y) maxY = pt[n][idx].y;
			}

			int count = 0;
			int tmp = 0;

			for (int i = 0; i < 10; ++i)
			{
				connectedNodeTerm[i] = -1;
			}

			for (int x = minX - 2; x < maxX + 2; ++x)
			{
				for (int y = minY - 2; y < maxY + 2; ++y)
				{
					int dx = MIN(MAX(x, 0), image->width - 1);
					int dy = MIN(MAX(y, 0), image->height - 1);

					if (FOV_clusters[dx][dy] == -1 || n == FOV_clusters[dx][dy])
						continue;
					else
					{
						if (count == 0)
						{
							connectedNodeTerm[count] = FOV_clusters[dx][dy];
							count++;
						}
						else
						{
							for (int c = 1; c < 10; ++c)
							{
								if (connectedNodeTerm[c - 1] == FOV_clusters[dx][dy])
									break;
								else
								{
									if (connectedNodeTerm[c] != -1)
										continue;
									else
									{
										connectedNodeTerm[c] = FOV_clusters[dx][dy];
										break;
									}
								}
							}
						}
					}
				}
			}

			vector<int> connectedTempNode;
			for (int t = 0; t < 10; ++t)
			{
				if (connectedNodeTerm[t] != -1)
					connectedTempNode.push_back(connectedNodeTerm[t]);
			}
			connectedNode.push_back(connectedTempNode);
		}

		if (i != NR_ITERATIONS - 1)
			connectedNode.clear();
	}
}

/*
* Enforce connectivity of the superpixels. This part is not actively discussed
* in the paper, but forms an active part of the implementation of the authors
* of the paper.
*
* Input : The image (IplImage*).
* Output: -
*/
//아직 이부분 이해가 안가는데,, 결론은 cluster에 포함되지 않은 픽셀을 인접 클러스터로 할당.
void Slic::create_connectivity(IplImage* image) {

	int label = 0, adjlabel = 0;
	const int lims = (image->width * image->height) / ((int)FOV_centers.size());

	const int dx4[4] = { -1, 0, 1, 0 };
	const int dy4[4] = { 0, -1, 0, 1 };

	/* Initialize the new cluster matrix. */
	vec2di FOV_new_clusters;
	for (int i = 0; i < image->width; i++) {
		vector<int> nc;
		for (int j = 0; j < image->height; j++) {
			nc.push_back(-1);
		}
		FOV_new_clusters.push_back(nc);
	}

	for (int i = 0; i < image->width; i++) {
		for (int j = 0; j < image->height; j++) {

			int xx = (i - (image->width / 2)) * (i - (image->width / 2));
			int yy = (j - (image->height / 2)) * (j - (image->height / 2));
			int rr = (image->width / 2) * (image->width / 2);
			if (xx + yy <= rr)
			{
				if (FOV_new_clusters[i][j] == -1) {
					vector<CvPoint> elements;
					elements.push_back(cvPoint(i, j));

					/* Find an adjacent label, for possible use later. */
					for (int k = 0; k < 4; k++) {
						int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];

						if (x >= 0 && x < image->width && y >= 0 && y < image->height) {
							if (FOV_new_clusters[x][y] >= 0) {
								adjlabel = FOV_new_clusters[x][y];
							}
						}
					}

					int count = 1;
					for (int c = 0; c < count; c++) {
						for (int k = 0; k < 4; k++) {
							int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];

							if (x >= 0 && x < image->width && y >= 0 && y < image->height) {
								if (FOV_new_clusters[x][y] == -1 && FOV_clusters[i][j] == FOV_clusters[x][y]) {
									elements.push_back(cvPoint(x, y));
									FOV_new_clusters[x][y] = label;
									count += 1;
								}
							}
						}
					}

					/* Use the earlier found adjacent label if a segment size is
					smaller than a limit. */
					if (count <= lims >> 2) {
						for (int c = 0; c < count; c++) {
							FOV_new_clusters[elements[c].x][elements[c].y] = adjlabel;
						}
						label -= 1;
					}
					label += 1;
				}
			}
		}
	}
}

/*
* Display the cluster centers.
*
* Input : The image to display upon (IplImage*) and the colour (CvScalar).
* Output: -
*/
void Slic::display_center_grid(IplImage* image, CvScalar colour) {
	for (int i = 0; i < (int)FOV_centers.size(); i++) {
		cvCircle(image, cvPoint(FOV_centers[i][3], FOV_centers[i][4]), 2, colour, 2);
	}
}

/*
* Display a single pixel wide contour around the clusters.
*
* Input : The target image (IplImage*) and contour colour (CvScalar).
* Output: -
*/
void Slic::display_contours(IplImage* image, CvScalar colour) {
	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	/* Initialize the contour vector and the matrix detailing whether a pixel
	* is already taken to be a contour. */
	vector<CvPoint> contours;
	vec2db istaken;
	for (int i = 0; i < image->width; i++) {
		vector<bool> nb;
		for (int j = 0; j < image->height; j++) {
			nb.push_back(false);
		}
		istaken.push_back(nb);
	}

	/* Go through all the pixels. */
	for (int i = 0; i < image->width; i++) {
		for (int j = 0; j < image->height; j++) {
			int nr_p = 0;

			int xx = (i - (image->width / 2)) * (i - (image->width / 2));
			int yy = (j - (image->height / 2)) * (j - (image->height / 2));
			int rr = (image->width / 2) * (image->width / 2);
			if (xx + yy <= rr)
			{
				/* Compare the pixel to its 8 neighbours. */
				for (int k = 0; k < 8; k++) {
					int x = i + dx8[k], y = j + dy8[k];

					if (x >= 0 && x < image->width && y >= 0 && y < image->height) {
						if (istaken[x][y] == false && FOV_clusters[i][j] != FOV_clusters[x][y]) {
							nr_p += 1;
						}
					}
				}

				/* Add the pixel to the contour list if desired. */
				if (nr_p >= 2) {
					contours.push_back(cvPoint(i, j));
					istaken[i][j] = true;
				}
			}
		}
	}

	/* Draw the contour pixels. */
	for (int i = 0; i < (int)contours.size(); i++) {
		cvSet2D(image, contours[i].y, contours[i].x, colour);
	}
}

/*
* Give the pixels of each cluster the same colour values. The specified colour
* is the mean RGB colour per cluster.
*
* Input : The target image (IplImage*).
* Output: -
*/
void Slic::colour_with_cluster_means(IplImage* image) {


	vector<CvScalar> FOV_colours(FOV_centers.size());
	vector<double> FOV_m_colours;


	/* Gather the colour values per cluster. */
	for (int i = 0; i < image->width; i++) {
		for (int j = 0; j < image->height; j++) {

			int index = FOV_clusters[i][j];
		
			CvScalar colour = cvGet2D(image, j, i);

			int xx = (i - (image->width / 2)) * (i - (image->width / 2));
			int yy = (j - (image->height / 2)) * (j - (image->height / 2));
			int rr = (image->width / 2) * (image->width / 2);
			if (xx + yy <= rr)
			{

				FOV_colours[index].val[0] += colour.val[0];
				FOV_colours[index].val[1] += colour.val[1];
				FOV_colours[index].val[2] += colour.val[2];
			}
		}
	}

	/* Divide by the number of pixels per cluster to get the mean colour. */
	for (int i = 0; i < (int)FOV_colours.size(); i++) {
		FOV_colours[i].val[0] /= FOV_center_counts[i];
		FOV_colours[i].val[1] /= FOV_center_counts[i];
		FOV_colours[i].val[2] /= FOV_center_counts[i];

		avg_colors[i].val[0] /= FOV_center_counts[i];
		avg_colors[i].val[1] /= FOV_center_counts[i];
		avg_colors[i].val[2] /= FOV_center_counts[i];
	}

	/* Fill in. */
	for (int i = 0; i < image->width; i++) {
		for (int j = 0; j < image->height; j++) {

			int xx = (i - (image->width / 2)) * (i - (image->width / 2));
			int yy = (j - (image->height / 2)) * (j - (image->height / 2));
			int rr = (image->width / 2) * (image->width / 2);
			if (xx + yy <= rr)
			{
				CvScalar ncolour = FOV_colours[FOV_clusters[i][j]];
				cvSet2D(image, j, i, ncolour);

				newImage.at<cv::Vec3b>(j, i) = (cv::Vec3b)avg_colors[FOV_clusters[i][j]];
			}
		}

	}

	//// SP_centers generate
	//for (int i = 0; i < (int)FOV_colours.size(); i++) {
	//	FOV_m_colours.push_back(FOV_colours[i].val[0]);
	//	FOV_m_colours.push_back(FOV_colours[i].val[1]);
	//	FOV_m_colours.push_back(FOV_colours[i].val[2]);

	//	FOV_SP_centers.push_back(FOV_m_colours);

	//	FOV_m_colours.clear();
	//}
}