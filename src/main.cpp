//
// Created by Gabriel on 07/06/23.
//

/*
int main() {


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
        break;
        Mat image1;
        Mat image = detectDishEdge(tray6[i]);
        cvtColor(image,image,COLOR_BGRA2GRAY);
        double thres = threshold(image,image1,0,190,THRESH_OTSU);

        //Mat image = tray1[i];
        //Mat imageCircles = detectDishEdge(image);

        //imshow("Dish",image);
        imshow("dish1",image1);
        waitKey(0);
    }

        //Mat image1 = detectDishEdge(tray1[0]);
        float factor = 0.8;
      //  Mat augmented_image = augmentation(image1, factor);
        //imshow("a",augmented_image);
        //waitKey(0);


    /*Mat image = tray2[0];
    Mat temp = detectSalad(image);
    imshow("temp", temp);
    waitKey(0);*/
    //Mat dishes = detectDishEdge(image1);




    //Mat image = detectDishEdge(tray1[0]);
    //Mat processed = preprocessImage(dishes);

    //drawContours(beforeMealProcessed,beforeMealContours,-1,Scalar(0,0,255),2);
    //imshow("qqq",processed);
    //waitKey(0);



    /*// Converti l'immagine da BGR a HSV
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    // Definisci i range di colore per il piatto (es. marrone)
    Scalar lowerBound(0, 0, 66);
    Scalar upperBound(255, 30, 255);

    // Crea una maschera per il piatto
    Mat mask;
    inRange(hsvImage, lowerBound, upperBound, mask);

    // Applica la maschera all'immagine originale per ottenere solo il cibo
    Mat result;
    bitwise_and(image, image, result, mask);


    */

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>
#include "detectDish.h"
#include "utils.h"

using namespace cv;
using namespace std;


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


    for(int i=0;i<4;i++)
    {
        Mat img = detectDishesEdge(tray2[i]);
        int delta = 20;

        Mat salad = detectSalad(tray2[i]);
        Mat temp = removeDishes(img, delta);
        imshow("temp", temp);
        imshow("salad",salad);
        waitKey(0);


    }

    return 0;
}