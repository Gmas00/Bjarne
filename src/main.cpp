//
// Created by Gabriel on 07/06/23.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "detectFood.h"

using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("../src/resource/Food_leftover_dataset/tray1/leftover1.jpg");
    Mat ImageCircles = detectDishEdge(image);
    imshow("Dish",ImageCircles);
    waitKey(0);
    return 0;
}





