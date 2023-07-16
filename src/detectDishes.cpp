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

//altra funzione per il pane
Mat detectBreadByHisto(const Mat& image)
{
    Mat external, temp;
    temp = detectFoods(image);
    external = image - temp;
    Mat bread;
    int size = 280;
    int numColors = 280;
    int delta = 28;
    int thresold = 85;
    bread = removeDishes(external,25);
    //test, scommenta e usi istogramma
    //bread = removeColors(external,size,numColors,delta);
    Scalar targetColor1(38, 187, 181);
    Scalar targetColor2(2,53,73);
    Scalar targetColor3(3,118,121);
    Scalar targetColor4(84,124,153);
    Scalar targetColor5(21,165,160);
    Scalar targetColor6(255,255,255);
    Scalar targetColor7(2,215,206);
    Scalar targetColor8(40,92,186);
    Scalar targetColor9(231,208,192);

    removeSimilarPixels(bread,targetColor1,thresold);
    removeSimilarPixels(bread,targetColor2,thresold);
    removeSimilarPixels(bread,targetColor3,thresold);
    removeSimilarPixels(bread,targetColor4,thresold);
    removeSimilarPixels(bread,targetColor5,thresold);
    removeSimilarPixels(bread,targetColor6,thresold);
    removeSimilarPixels(bread,targetColor7,thresold);
    removeSimilarPixels(bread,targetColor8,thresold);
    removeSimilarPixels(bread,targetColor9,thresold);
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

//Fanculo queste 3 funzioni non funzionano
//Hough transform to detect lines, evaluate the angles between them and detect squares
void findSquares(const Mat& image, vector<vector<Point>>& squares){
    int thresh = 50, N=11;

    squares.clear();
    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    //Down and up scale image to filter noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point>> contours;

    //Find squares in every color plane of the image
    for(int c=0; c<3; c++){
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        //Try several threshold levels
        for(int l=0; l<N; l++){
            //Use Canny instead of zero threshold level to catch squares with gradient 
            if(l==0){
                Canny(gray0, gray, 0, thresh, 5);
                //Dilate Canny output to remove potential holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else{
                //Apply threshold if l!=0: tgray(x,y)=gray(x,y)<(l+1)*255/N ? 255:0
                gray = gray0 >= (l+1)*255/N;
            }

            //Find contours and store them in a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            //Test contours
            for(size_t i=0; i<contours.size(); i++){
                //Approximate countour with accuracy proportional to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                //Square contours should have 4 vertices after approximation, be convex and have area > 1000
                //Area may be positive or negative, depending on contour orientation
                if(approx.size()==4 && fabs(contourArea(Mat(approx))) > 5 && isContourConvex(Mat(approx))){
                    double maxCosine=0;
                    for(int j=2; j<5; j++){
                        //Find maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    //If cosines of all angles are small then write quadrangle vertices to resultant sequence
                    if(maxCosine < 0.3)
                        squares.push_back(approx);

                }
            }

        }
    }
}

//Function to draw squares on the image
void drawSquares(Mat& image, const vector<vector<Point>>& squares, const char* wndname){
    for(size_t i=0; i<squares.size(); i++){
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }

    imshow(wndname, image);
}

//Helper function that finds cosine of angle between vectors
double angle(Point p1, Point p2, Point p0){
    double dx1 = p1.x - p0.x;
    double dy1 = p1.y - p0.y;
    double dx2 = p2.x - p0.x;
    double dy2 = p2.y - p0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}
/*
Mat HoughTransform(const Mat& image){
    Mat blur, canny, canny_bgr, cannyP;

    //Blurr the image
    GaussianBlur(image, blur, Size(9, 9), 0); 
    //Apply Canny edge detector
    Canny(blur, canny, 50, 200);

    //Copy edge image to another image to show BGR 
    cvtColor(canny, canny_bgr, COLOR_GRAY2BGR);
    cannyP = canny_bgr.clone();

    //Apply Standard Hough Line Transform
    vector<Vec2f> lines;
    HoughLines(canny, lines, 1, CV_PI / 180, 100, 0, 0, 0, CV_PI / 3);
    //Draw lines
    for(size_t i=0; i<lines.size(); i++){
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;

        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));

        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));

        line(canny_bgr, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }
}
*/