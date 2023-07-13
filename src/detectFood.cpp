//
// Created by Gabriel on 10/06/23.
//


#include "detectFood.h"
using namespace std;
using namespace cv;


Mat detectDishEdge(cv::Mat image)
{
    Mat ImageCircles;
    image.copyTo(ImageCircles);
    Mat gray;
    cvtColor(image,gray,COLOR_RGB2GRAY);

    medianBlur(gray,gray,5);

    //get the circle edges
    vector<Vec3f>circles;
    HoughCircles(gray,circles,HOUGH_GRADIENT,1,gray.rows/16,100,30,280,292);
    for (int i =0; i<circles.size();i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0],c[1]);
        circle(ImageCircles, center, 1,Scalar(0,100,100),3,LINE_AA);
        int radius = c[2];
        circle(ImageCircles,center,radius,Scalar(0,0,0),3,LINE_AA);
    }



    //mask to isolate only the dish
    /*Mat mask(ImageCircles.size(), CV_8UC1, Scalar(0));
    for (int i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        int radius = c[2];
        circle(mask, center, radius, Scalar(255), -1);
    }
    for(int i=0;i<ImageCircles.rows;i++)
    {
        for(int j=0;j<ImageCircles.cols;j++)
        {
            if(mask.at<unsigned char>(i,j)==0)
            {
                ImageCircles.at<Vec3b>(i,j) = 0;
            }
        }
    }*/

    return ImageCircles;
}
