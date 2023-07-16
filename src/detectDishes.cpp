//
// Created by Gabriel on 10/06/23.
//


#include "detectDishes.h"
#include "utils.h"
using namespace std;
using namespace cv;


//Function that given an image, returns a new image with only the dishes in the picture
Mat detectDishesEdge(const Mat& image)
{
    int max= 350;
    int min = 250;
    int hCanny = 100;
    int hCircle = 50;
    Mat imageCircles;
    image.copyTo(imageCircles);
    Mat gray;
    vector<Vec3f>circles;

    cvtColor(image,gray,COLOR_RGB2GRAY);

    medianBlur(gray,gray,7);

    Mat mask(imageCircles.size(), CV_8UC1, Scalar(0));
    //HoughCircles(gray,circles,HOUGH_GRADIENT,1,gray.rows/16,100,30,268,292);
   // HoughCircles(gray, circles, HOUGH_GRADIENT, 1,220,100, 20, 260, 280);
    HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, gray.rows / 16, hCanny, hCircle, min, max);
    for (int i =0; i<circles.size();i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0],c[1]);
        int radius = c[2];
        circle(mask, center, radius, Scalar(255), -1);
    }
    for(int i=0;i<imageCircles.rows;i++)
    {
        for(int j=0;j<imageCircles.cols;j++)
        {
            if(mask.at<unsigned char>(i,j)==0)
            {
                imageCircles.at<Vec3b>(i,j) = 0;
            }
        }
    }
    return imageCircles;
}
//Function that given an image, returns a new image with only the salad in the picture
Mat detectSalad(const Mat& image)
{
    int hCanny = 100;
    int hCircle = 50;
    int max = 210;
    int min = 175;
    Mat salad;
    Mat gray;
    vector<Vec3f> bowl;
    image.copyTo(salad);

    cvtColor(image,gray,COLOR_RGB2GRAY);

    HoughCircles(gray, bowl, HOUGH_GRADIENT, 1, gray.rows / 16, hCanny, hCircle, min, max);
    for (int i = 0; i < bowl.size(); i++)
        circle(salad, Point(bowl[i][0], bowl[i][1]), bowl[i][2], cv::Scalar(0, 0, 0), 2);

    Mat mask(image.size(), CV_8UC1,Scalar(0));
    for (int i =0; i<bowl.size();i++)
    {
        Vec3i c = bowl[i];
        Point center = Point(c[0],c[1]);
        // circle(ImageCircles, center, 1,Scalar(0,100,100),3,LINE_AA);
        int radius = c[2];
        //circle(ImageCircles,center,radius,Scalar(0,0,0),3,LINE_AA);
        circle(mask, center, radius, Scalar(255), -1);
    }

    for(int i=0;i<salad.rows;i++)
    {
        for(int j=0;j<salad.cols;j++)
        {
            if(mask.at<unsigned char>(i,j)==0)
            {
                salad.at<Vec3b>(i,j) = 0;
            }
        }
    }
    return salad;

}
//Function that given an image, returns a new image with dishes and salad in the picture
Mat detectFoods(const Mat& image)
{
    vector<Mat>images;
    Mat temp1, temp2;
    //image.copyTo(temp1);
    //image.copyTo(temp2);

    temp1 = detectDishesEdge(image);
    temp2 = detectSalad(image);
    Mat all = temp1 + temp2;
    return all;
}
//Function that given an image, returns a new image with only the bread in the picture
Mat detectBread(const Mat& image)
{
    const int THRESHOLD = 64;
    // Convert to HSV color space
    Mat hsv_image;
    cvtColor(image, hsv_image, COLOR_RGB2HSV);
    // Threshold the HSV image

    Mat thresholded_HSV;
    extractChannel(hsv_image, thresholded_HSV, 1);
    normalize(thresholded_HSV, thresholded_HSV, 0, 255, NORM_MINMAX);
    threshold(thresholded_HSV, thresholded_HSV, THRESHOLD, 255, THRESH_BINARY);

    // Filter the areas (to remove small objects and get only food blobs) DA RIVEDERE
    vector<vector<Point>> contours;
    findContours(thresholded_HSV, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for(int i=0; i<contours.size(); i++)
    {
        if(contourArea(contours[i]) > 8000)
        {
            drawContours(thresholded_HSV, contours, i, 255, -1);
        }
    }

    // Apply morphological operations to improve the mask
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(thresholded_HSV, thresholded_HSV, MORPH_CLOSE, kernel);
    //morphologyEx(thresholded_HSV, thresholded_HSV, MORPH_OPEN, kernel);

    // Apply max filter to remove small noise
    Mat max_filtered;
    dilate(thresholded_HSV, max_filtered, kernel);

    return max_filtered;
}




//ste
Mat removeDishes(Mat image, int delta)
{
    Mat img;
    image.copyTo(img);
    for(int i=0; i<img.rows; i++)
    {
        for(int j=0; j<img.cols; j++)
        {
            Vec3b pix = img.at<Vec3b>(i,j);
            int avg = (int)(pix[0] + pix[1] + pix[2])/3;
            if(pix[0]-avg < delta && pix[1]-avg < delta && pix[2]-avg < delta)
            {
                img.at<Vec3b>(i,j) [0]= 0;
                img.at<Vec3b>(i,j) [1]= 0;
                img.at<Vec3b>(i,j) [2] = 0;
            }
        }
    }
    return img;
}
