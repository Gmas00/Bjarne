//
// Created by eb0r0h on 07/06/23.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
    cout<<"Test"<<endl;

    Mat img(256,256,CV_8UC1);
    for(int i=0;i<256;i++)
    {
        for(int j=0;j<256;j++)
        {
            img.at<unsigned char>(i,j) = j;
        }
    }

    imshow("Gradient",img);
    waitKey(0);

    return 0;
}
