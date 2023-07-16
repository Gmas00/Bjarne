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
    
    Mat img, gray, blur, canny, cdstP;
    //Standard Hough Transform
    img=tray5[1];
    //Convert to grayscale
    cvtColor(img, gray, COLOR_BGR2GRAY);
    //Blur the image
    GaussianBlur(gray, blur, Size(9, 9), 0);
    //Detect edges with canny
    Canny(blur, canny, 10, 100, 3, true);

    //Copy edge image to another
    cvtColor(canny, cdstP, COLOR_GRAY2BGR);

    //Probabilistic Hough Transform
    vector<Vec4i> linesP;
    HoughLinesP(canny, linesP, 1, CV_PI / 180, 50, 50, 10);
    //Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        Vec4i l = linesP[i];
        //Draw the lines
        line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }

    imshow("Source", img);
    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);

    waitKey();
    



    /*
    vector<vector<Point>> squares;
    for(int i=0; i<tray5.size(); i++)
    {
        Mat img = tray5[i];
        findSquares(img, squares);
        drawSquares(img, squares, wndname);
        imshow(wndname, img);
        waitKey(0);
    }*/

    /*
    for(int i=0;i<4;i++)
    {
        
        Mat img = tray5[i];
        Mat bread = detectBreadByHisto(img);
        imshow("bread",bread);
        waitKey(0);

    }*/

//Mask m_10(tray1[3]);
    return 0;
}