//
// Created by eb0r0h on 15/07/23.
//


#include "utils.h"
using namespace std;
using namespace cv;

//Function that creates and returns a Mat vector, contains the images for each tray
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

//Function to compare colors, it is needed in findMostFrequentColors
bool compareColors(const pair<Vec3b, int>& color1, const pair<Vec3b, int>& color2)
{
    return color1.second > color2.second;
}

//Function that finds using the histogram the most frequent colors
vector<Vec3b> findMostFrequentColors(const Mat& image, int numColors)
{
    vector<Vec3b> mostFrequentColors;
    vector<Mat> channels;
    split(image, channels);

    int histSize = 256;
    float range[] = {1, 256};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    Mat b_hist, g_hist, r_hist;
    calcHist(&channels[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&channels[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&channels[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    vector<pair<Vec3b, int>> colorFrequencies;

    for (int i = 1; i < histSize; i++)
    {
        Vec3b color(i, i, i);
        int frequency = b_hist.at<float>(i) + g_hist.at<float>(i) + r_hist.at<float>(i);

        if (color != Vec3b(0, 0, 0)) // Escludi il colore nero
        {
            colorFrequencies.push_back(make_pair(color, frequency));
        }
    }

    sort(colorFrequencies.begin(), colorFrequencies.end(), compareColors);

    int colorsToSelect = min(numColors, static_cast<int>(colorFrequencies.size()));
    for (int i = 0; i < colorsToSelect; i++)
        mostFrequentColors.push_back(colorFrequencies[i].first);

    return mostFrequentColors;
}

//Function that removes the most frequent colors from the image
Mat removeColors(const Mat& image1,int size,int numColors,int delta)
{
    Mat image;
    image1.copyTo(image);
    //size = 50
    for(int i=0;i<size;i++)
    {
        //numcolors =100;
        vector<Vec3b> mostFrequentColors = findMostFrequentColors(image, numColors);
        Vec3b targetColor = mostFrequentColors[i];
        //delta=22
        for (int y = 0; y < image.rows; y++)
        {
            for (int x = 0; x < image.cols; x++)
            {
                Vec3b currentColor = image.at<Vec3b>(y, x);
                if (abs(currentColor[0] - targetColor[0]) <= delta && abs(currentColor[1] - targetColor[1]) <= delta && abs(currentColor[2] - targetColor[2]) <= delta)
                    image.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
            }
        }
    }
    return image;
}

//Function that returns a vector with all the labels
vector<string> getLabels()
{
    vector<string> labels = {
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
    return labels;

}







/*
void filterAreas(const cv::Mat& input,Mat& output,int threshold)
{

    //Mat gray;
    //cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    vector<vector<Point>> contours;
    findContours(input, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++)
    {
        if (contourArea(contours[i]) > threshold)
        {
            drawContours(output, contours, i, 255, -1);
        }
    }
}*/

