#include "slic.hpp"

int main()
{
	cv::Mat img = cv::imread("Input_Image/subImage4.png");
	if (img.empty())
		return -1;

	cv::imshow("srcImage", img);

	cv::blur(img, img, cv::Size(3, 3));
	cv::imshow("Blurred", img);
	SLIC slic(img, 100, 2000);

	cv::Mat mean_img = slic.colour_with_cluster_means();
	cv::imshow("mean_img",mean_img);
	cv::imwrite("superpixel_Image4.png", mean_img);
	cv::waitKey();
	cv::Mat contour = slic.display_contours(cv::Vec3b(0, 0, 255));
	cv::imshow("contour", contour);
	cv::imwrite("superpixel_Image_contour4.png", contour);
	cv::waitKey();

	return 0;
}