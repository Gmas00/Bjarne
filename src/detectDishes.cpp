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
    temp1 = detectDishesEdge(image);
    temp2 = detectSalad(image);
    Mat all = temp1 + temp2;
    return all;
}


//Function that given an image, returns a new image with only the bread in the picture
// AJ CODE, serve nella funzione dopo probabilmente
Mat detectBread(const Mat& image)
{
    const int THRESHOLD = 32;
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
        if(contourArea(contours[i]) > 4000)
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

//altra funzione per il pane, forse il secondo parametro nn serve
Mat detectBreadByHisto(const Mat& image, Mat& originalImage)
{
    Mat bread;
    int size = 50;
    int numColors = 100;
    int delta = 30;
    bread = removeColors(image,size,numColors,delta);
    //Mat temp = detectFoods(originalImage);

    Scalar targetColor1(38, 187, 181);
    Scalar targetColor2(2,53,73);
    Scalar targetColor3(197,95,43);     //Arancione-marroncino del biglietto dessert
    Scalar targetColor4(213,213,0);     //Post it giallo
    Scalar targetColor5(0,62,182);      //Blu dello yogurt
    Scalar targetColor6(0,10,87);       //Blu scuro dello yogurt
    Scalar targetColor7(210,120,60);    //Un altro Arancione del biglietto dessert
    Scalar targetColor8(0,59,153);      //Un altro blu dello yogurt
    //rgb(73,53,2)
    //aggiungere altri target, così becco il biglietto giallo
    //uso estensione su chrome per ottenere rbg

    int thresold = 100;
    removeSimilarPixels(bread,targetColor1,thresold);
    removeSimilarPixels(bread,targetColor2,thresold);
    removeSimilarPixels(bread,targetColor3,thresold);
    removeSimilarPixels(bread,targetColor4,thresold);
    removeSimilarPixels(bread,targetColor5,thresold);
    removeSimilarPixels(bread,targetColor6,thresold);
    removeSimilarPixels(bread,targetColor7,thresold);
    removeSimilarPixels(bread,targetColor8,thresold);

    //bread = bread - temp;
    return bread;
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
                img.at<Vec3b>(i,j) [0] = 0;
                img.at<Vec3b>(i,j) [1] = 0;
                img.at<Vec3b>(i,j) [2] = 0;
            }
        }
    }
    return img;
}
