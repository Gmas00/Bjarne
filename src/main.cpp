//
// Created by Gabriel on 10/06/23.
//


#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "detectDishes.h"
#include "utils.h"

using namespace cv;
using namespace std;




int main()
{
    const vector<string> labels = getLabels();

    vector<Mat> tray1, tray2, tray3, tray4, tray5, tray6, tray7, tray8;
    tray1 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray1/");
    tray2 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray2/");
    tray3 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray3/");
    tray4 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray4/");
    tray5 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray5/");
    tray6 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray6/");
    tray7 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray7/");
    tray8 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray8/");

    for(int i=0;i<4;i++)
    {
        Mat img = tray5[i];
        Mat external, temp;
        temp = detectFoods(img);
        external = img - temp;

        Mat image;
        external.copyTo(image);
        Mat bread_hist = detectBreadByHisto(image,img);
        Mat bread = detectBread(image);

        //imshow("Bread histogram", bread_hist);
        imshow("Bread", bread);
        Mat img = tray4[i];
        Mat bread = detectBreadByHisto(img);
        imshow("bread",bread);
        waitKey(0);

    }

//Mask m_10(tray1[3]);
    return 0;
}