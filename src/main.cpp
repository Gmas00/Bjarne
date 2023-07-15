//
// Created by Gabriel on 10/06/23.
//


#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "detectDishes.h"
#include "utils.h"
#include "Mask.h"

using namespace cv;
using namespace std;


int main()
{
    const vector<string> labels ={
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


    //Mask mas(tray2[0]);


    return 0;
}