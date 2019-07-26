#include "slic.hpp"


SLIC::SLIC(cv::Mat& img, float M, float k)
{
	this->img = img.clone();
	this->M = M;
	this->w = img.cols;
	this->h = img.rows;
	if (k == USE_DEFAULT_S)
	{
		this->step = (img.cols / 15 + img.rows / 15 + 1) / 2;
	}

	this->step = sqrt((img.cols * img.rows) / k);

	generateSuperpixels();

}

void SLIC::init_data()
{
	// Scale img to [0,1]
	this->img.convertTo(this->img_f, CV_32F, 1 / 255.);

	// Convert to l-a-b colorspace
	cv::cvtColor(this->img_f, this->img_lab, CV_BGR2Lab);

	this->labels = -1 * cv::Mat::ones(this->img_lab.size(), CV_32S);
	this->distances = FLT_MAX * cv::Mat::ones(this->img_lab.size(), CV_32F);


	//int step = this->step;
	//int w = this->w, h = this->h;

	for (int i = step; i < w - step / 2; i += step)
		for (int j = step; j < h - step / 2; j += step)
		{
			vector<float> FOV_center;
			/* Find the local minimum (gradient-wise) */
			cv::Point c = find_local_minimum(cv::Point(i, j));
			cv::Vec3f colour = this->img_lab.at<cv::Vec3f>(c.y, c.x);

			int x = (i - (w / 2)) * (i - (w / 2));
			int y = (j - (h / 2)) * (j - (h / 2));
			int r = (w / 2) * (w / 2);
			if (x + y <= r)
			{
				FOV_center.push_back(colour.val[0]);
				FOV_center.push_back(colour.val[1]);
				FOV_center.push_back(colour.val[2]);

				centers_value.push_back(FOV_center);
				centers.push_back(c);
				center_counts.push_back(0);
			}

		}

}


cv::Point SLIC::find_local_minimum(cv::Point c)
{
	float min_grad = FLT_MAX;
	cv::Point loc_min = cv::Point(c.x, c.y);

	for (int i = c.x - 1; i < c.x + 2; i++) {
		for (int j = c.y - 1; j < c.y + 2; j++)
		{
			cv::Vec3f c1 = this->img_lab.at<cv::Vec3f>(j + 1, i);
			cv::Vec3f c2 = this->img_lab.at<cv::Vec3f>(j, i + 1);
			cv::Vec3f c3 = this->img_lab.at<cv::Vec3f>(j, i);

			// Convert colour values to grayscale values
			float i1 = c1.val[0];
			float i2 = c2.val[0];
			float i3 = c3.val[0];

			if (sqrt(pow(i1 - i3, 2)) + sqrt(pow(i2 - i3, 2)) < min_grad) {
				min_grad = fabs(i1 - i3) + fabs(i2 - i3);
				loc_min.x = i;
				loc_min.y = j;
			}
		}
	}
	return loc_min;
}

float SLIC::dist(int ci, cv::Point pixel, cv::Vec3f colour)
{
	float dc = sqrt(pow(centers_value[ci][0] - colour.val[0], 2) + pow(centers_value[ci][1]
		- colour.val[1], 2) + pow(centers_value[ci][2] - colour.val[2], 2));
	float ds = sqrt(pow(centers[ci].x - pixel.x, 2) + pow(centers[ci].y - pixel.y, 2));

	return dc + M / step * ds;
}



void SLIC::generateSuperpixels()
{
	init_data();

	/* Run EM for 10 iterations (as prescribed by the algorithm) */
	for (int i = 0; i < NR_ITERATIONS; i++) {
		/* Reset distance values. */
		this->distances = FLT_MAX * cv::Mat::ones(this->img_lab.size(), CV_32F);

		for (int j = 0; j < (int)centers_value.size(); j++) {
			for (int k = centers[j].x - step; k < centers[j].x + step; k++) // 2S사이 x
				for (int l = centers[j].y - step; l < centers[j].y + step; l++) { //2S사이 y
					
					int xx = (k - w / 2) * (k - w / 2);
					int yy = (l - h / 2) * (l - h / 2);
					int r = (w / 2) * (w / 2);
					if (xx + yy <= r) {
						if (k >= 0 && k < w && l >= 0 && l < h) {
							cv::Vec3f colour = this->img_lab.at<cv::Vec3f>(l, k);
							float d = dist(j, cv::Point(k, l), colour);

							/* Update cluster allocation if the cluster minimizes the distance */
							if (d < distances.at<float>(l, k))
							{
								distances.at<float>(l, k) = d;
								labels.at<int>(l, k) = j;
							}
						}
					}
				}

		}

		/* Clear the center values. */
		for (int j = 0; j < (int)centers_value.size(); j++) {

			centers[j].x = 0; centers[j].y = 0;
			centers_value[j][0] = 0;
			centers_value[j][1] = 0;
			centers_value[j][2] = 0;
			center_counts[j] = 0;
		}

		/* Compute the new cluster centers. */
		for (int j = 0; j < w; j++) {
			for (int k = 0; k < h; k++) {
				int c_id = labels.at<int>(k, j);

				int x = (j - (w / 2)) * (j - (w / 2));
				int y = (k - (h / 2)) * (k - (h / 2));
				int r = (w / 2) * (w / 2);
				if (x + y <= r)
				{
					if (c_id != -1) {
						cv::Vec3f colour = img_lab.at<cv::Vec3f>(k, j);
						centers_value[c_id][0] += colour.val[0];
						centers_value[c_id][1] += colour.val[1];
						centers_value[c_id][2] += colour.val[2];
						centers[c_id].x = j;
						centers[c_id].y = k;
						center_counts[c_id] += 1;
					}
				}
			}
		}

		/* Normalize the clusters. */
		for (int j = 0; j < (int)centers_value.size(); j++) {

			if (center_counts[j] == 0)
				center_counts[j] = 1;

			centers_value[j][0] /= center_counts[j];
			centers_value[j][1] /= center_counts[j];
			centers_value[j][2] /= center_counts[j];
			centers[j].x /= center_counts[j];
			centers[j].y /= center_counts[j];

		}
	}
}

cv::Mat SLIC::display_contours(cv::Vec3b colour)
{
	//this->show = this->img.clone();
	const int dx8[8] = { -1,-1,0,1,1,1,0,-1 };
	const int dy8[8] = { 0,-1,-1,-1,0,1,1,1 };

	/* Initialize the contour vector and the matrix detailing whether a pixel is already taken to be a contour. */
	vector<cv::Point> contours;
	vector<vector<bool>> istaken;
	for (int i = 0; i < w; i++) {
		vector<bool>nb;
		for (int j = 0; j < h; j++) {
			nb.push_back(false);
		}
		istaken.push_back(nb);
	}

	/* Go through all the pixels. */
	for (int i = 0; i < w; i++)
		for (int j = 0; j < h; j++) {
			int nr_p = 0;

			int x = (i - (w / 2)) * (i - (w / 2));
			int y = (j - (h / 2)) * (j - (h / 2));
			int r = (w / 2) * (w / 2);
			if (x + y <= r) {
				/* Compare the pixel to its 8 neighbours. */
				for (int k = 0; k < 8; k++) {
					int x = i + dx8[k], y = j + dy8[k];

					if (x >= 0 && x < w && y >= 0 && y < h) {
						if (istaken[x][y] == false && labels.at<int>(j, i) != labels.at<int>(y, x))
							nr_p += 1;
					}
				}
				if (nr_p >= 2) {
					contours.push_back(cv::Point(i, j));
					istaken[i][j] = true;
				}
			}

		}

	/* Draw the contour pixels. */
	for (int i = 0; i < (int)contours.size(); i++) {

		this->show.at<cv::Vec3b>(contours[i].y, contours[i].x) = colour;
	}

	return this->show.clone();
}

cv::Mat SLIC::colour_with_cluster_means() {

	int n = centers.size();
	vector<cv::Vec3b> colours(n);
	vector<long> B(n);
	vector<long> G(n);
	vector<long> R(n);
	vector<int> num_pixels(n);

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			int x = (i - (w / 2)) * (i - (w / 2));
			int y = (j - (h / 2)) * (j - (h / 2));
			int r = (w / 2) * (w / 2);
			if (x + y <= r) {
				int index = labels.at<int>(j, i);
				cv::Vec3b colour = img.at<cv::Vec3b>(j, i);

				B[index] += (int)colour[0];
				G[index] += (int)colour[1];
				R[index] += (int)colour[2];
				++num_pixels[index];
			}
		}
	}
	for (int i = 0; i < n; i++) {
		int num = num_pixels[i];
		if (num == 0)
			num = 1;
		colours[i] = cv::Vec3b(B[i] / num, G[i] / num, R[i] / num);
	}
	this->show = this->img.clone();
	for (int i = 0; i < w; i++)
		for (int j = 0; j < h; j++) {
			int x = (i - (w / 2)) * (i - (w / 2));
			int y = (j - (h / 2)) * (j - (h / 2));
			int r = (w / 2) * (w / 2);
			if (x + y <= r) {
				int L = labels.at<int>(j, i);
				if (center_counts[L])
					show.at<cv::Vec3b>(j, i) = colours[L];
			}
		}
	return show.clone();
}


vector<cv::Point> SLIC::getCenters() {
	return centers;
}

cv::Mat SLIC::getLabels() {
	return labels;
}

