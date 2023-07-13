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


    vector<Mat> tray1, tray2, tray3, tray4, tray5, tray6, tray7, tray8;
    tray1 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray1/");
    tray2 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray2/");
    tray3 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray3/");
    tray4 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray4/");
    tray5 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray5/");
    tray6 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray6/");
    tray7 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray7/");
    tray8 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray8/");


    for(int i = 0;i<4;i++)
    {
        Mat image = tray8[i];
        Mat ImageCircles = detectDishEdge(image);
        imshow("Dish",ImageCircles);
        waitKey(0);

    }





    return 0;

}





