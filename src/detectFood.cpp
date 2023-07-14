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
    HoughCircles(gray,circles,HOUGH_GRADIENT,1,gray.rows/16,100,30,270,292);
    for (int i =0; i<circles.size();i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0],c[1]);
        circle(ImageCircles, center, 1,Scalar(0,100,100),3,LINE_AA);
        int radius = c[2];
        circle(ImageCircles,center,radius,Scalar(0,0,0),3,LINE_AA);
    }

    //mask to isolate only the dish
    Mat mask(ImageCircles.size(), CV_8UC1, Scalar(0));
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
    }
    return ImageCircles;
}

vector<Mat> createVecImgFromSource(string path)
{
    vector<Mat> images;
    vector<string> fileNames;
    glob(path, fileNames);
    for (const auto& filename : fileNames)
    {
        Mat image = imread(filename);
        images.push_back(image);
    }
    return images;

}

Mat watershedByOpencCV(Mat src)
{
    Mat mask;
    inRange(src, Scalar(255, 255, 255), Scalar(255, 255, 255), mask);
    src.setTo(Scalar(0, 0, 0), mask);
// Show output image
//imshow("Black Background Image", src);
// Create a kernel that we will use to sharpen our image
    Mat kernel = (Mat_<float>(3, 3) <<
                                    1, 1, 1,
            1, -8, 1,
            1, 1, 1); // an approximation of second derivative, a quite strong kernel
// do the laplacian filtering as it is
// well, we need to convert everything in something more deeper then CV_8U
// because the kernel has some negative values,
// and we can expect in general to have a Laplacian image with negative values
// BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
// so the possible negative number will be truncated
    Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
// convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
// imshow( "Laplace Filtered Image", imgLaplacian );
//imshow( "New Sharped Image", imgResult );
// Create binary image from source image
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
//imshow("Binary Image", bw);
// Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);
// Normalize the distance image for range = {0.0, 1.0}
// so we can visualize and threshold it
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
//imshow("Distance Transform Image", dist);
// Threshold to obtain the peaks
// This will be the markers for the foreground objects
    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
// Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
//imshow("Peaks", dist);
// Create the CV_8U version of the distance image
// It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
// Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
// Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32S);
// Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
    }
// Draw the background marker
    circle(markers, Point(5, 5), 3, Scalar(255), -1);
    Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);
//imshow("Markers", markers8u);
// Perform the watershed algorithm
    watershed(imgResult, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);

    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++) {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar) b, (uchar) g, (uchar) r));
    }
// Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
// Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size())) {
                dst.at<Vec3b>(i, j) = colors[index - 1];
            }
        }
    }

    return dst;
}

//tentativo ma bocciato
Mat augmentation(Mat image0, float factor)
{
    Mat image;
    image0.copyTo(image);
    Mat hsv_image;
    cvtColor(image, hsv_image, COLOR_RGB2HSV);

    vector<Mat> channels;
    split(hsv_image, channels);

    Mat h = channels[0];
    Mat s = channels[1];
    Mat v = channels[2];

    // Fact * channel
    h = h * factor;
    s = s * factor;
    v = v * factor;

    // Clamp values to [0, 255]
    threshold(h, h, 255, 255, THRESH_TRUNC);
    threshold(s, s, 255, 255, THRESH_TRUNC);
    threshold(v, v, 255, 255, THRESH_TRUNC);

    // Merge channels
    Mat augmented_hsv;
    merge(channels, augmented_hsv);

    Mat augmented_rgb;
    cvtColor(augmented_hsv, augmented_rgb, COLOR_HSV2RGB);

    // Convert to grayscale
    Mat gray;
    cvtColor(augmented_rgb, gray, COLOR_RGB2GRAY);

    // Apply threshold to isolate colored areas
    Mat mask;
    threshold(gray, mask, 1, 255, THRESH_BINARY);

    // Set non-colored areas to black
    Mat result = augmented_rgb.clone();
    result.setTo(Scalar(0, 0, 0), ~mask);

    return result;
}

Mat segmentationHope(Mat dishes0)
{
    Mat dishes;
    dishes0.copyTo(dishes);
    Mat blurredImage;
    GaussianBlur(dishes, blurredImage, Size(5,5), 0);

    Mat hsvImage;
    cvtColor(blurredImage, hsvImage, COLOR_BGR2HSV);

    Scalar lowerThreshold = Scalar(0, 30, 60);
    Scalar upperThreshold = Scalar(30, 255, 255);

    Mat mask;
    inRange(hsvImage, lowerThreshold, upperThreshold, mask);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourImage = Mat::zeros(dishes.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area > 2000)
        {
            drawContours(contourImage, contours, static_cast<int>(i), Scalar(255, 255, 255), FILLED);
        }
    }

    cvtColor(contourImage,contourImage,COLOR_HSV2BGR);
    cvtColor(contourImage,contourImage,COLOR_BGRA2GRAY);

    for(int i=0;i<dishes.rows;i++)
    {
        for(int j=0;j<dishes.cols;j++)
        {
            if(contourImage.at<unsigned char>(i,j)==0)
            {
                dishes.at<Vec3b>(i,j)[0]=0;
                dishes.at<Vec3b>(i,j)[1]=0;
                dishes.at<Vec3b>(i,j)[2]=0;
            }
        }
    }

    return  dishes;
}

Mat detectSalad(cv::Mat image)
{
    Mat ImageCircles;
    image.copyTo(ImageCircles);
    Mat gray;
    cvtColor(image,gray,COLOR_RGB2GRAY);

    medianBlur(gray,gray,5);

    //get the circle edges
    vector<Vec3f>circles;
    HoughCircles(gray,circles,HOUGH_GRADIENT,1,gray.rows/16,100,30,180,200);
    for (int i =0; i<circles.size();i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0],c[1]);
        circle(ImageCircles, center, 1,Scalar(0,100,100),3,LINE_AA);
        int radius = c[2];
        circle(ImageCircles,center,radius,Scalar(0,0,0),3,LINE_AA);
    }

    //mask to isolate only the dish
    Mat mask(ImageCircles.size(), CV_8UC1, Scalar(0));
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
    }
    return ImageCircles;
}