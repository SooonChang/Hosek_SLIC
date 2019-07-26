#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <string>

#include "Superpixels.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){


    Mat img = imread("dog.png");

    if(!img.data){
        cout << "Bad image ..." << endl;
        return 0;
    }
    
    imshow("Original", img);

    blur(img, img, Size(3,3));
    imshow("Blurred", img);
    
    Superpixels sp(img);
    
    Mat boundaries = sp.viewSuperpixels();
    imshow("Superpixel boundaries", boundaries);
    
    Mat recolored = sp.colorSuperpixels();
    imshow("Average superpixel colors", recolored);


    // centers of the K clusters
    vector<Point> centers = sp.getCenters();
    
    waitKey(0);

    return 0;
}
