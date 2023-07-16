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


void filterAreas(const cv::Mat& input, cv::Mat& output, int threshold)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(input.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours)
    {
        if (cv::contourArea(contour) > threshold)
        {
            cv::drawContours(output, { contour }, -1, 255, cv::FILLED);
        }
    }
}



int main()
{
    const vector<string> labels = {
            "Background",
            "pasta with pesto",
            "pasta with tomato sauce",
            "pasta with meat sauce",
            "pasta with clams and mussels",
            "pilaw rice with peppers and peas",
            "grilled pork cutlet",
            "fish cutlet",
            "rabbit",
            "seafood salad",
            "beans",
            "basil potatoes",
            "salad",
            "bread"
    };


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
        Mat image, output;
        image = tray8[i];
        output = detectFoods(image);
        imshow("food", output);
        waitKey(0);
    }

    Mat temp2;
    Mat img = tray1[2];
    Mat external, temp;
    temp = detectFoods(img);
    external = img - temp;
    //imshow("ex",external);
    //waitKey(0);

    Mat bread = detectBread(img);
    //imshow("bread", bread);
    //waitKey(0);


    //int AREA_THRESHOLD_1 = 4000;
    //int AREA_THRESHOLD_2 = 8000;
    //int CONTOURS_DISTANCE_THRESHOLD = 45;


    //Mask m_10(tray2[0]);



    return 0;
}