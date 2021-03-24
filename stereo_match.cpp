#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace cv;
using namespace std;

int main() {
    for (int i = 0; i <=2; i += 2){

        Mat hand = imread("/Users/ccfeng/Desktop/data/data/paper_exp/Fig.4/hand/noisy/" + std::to_string(i+1) + ".bmp", 0);
        Mat pattern = imread("/Users/ccfeng/Desktop/data/data/paper_exp/Fig.4/hand/noisy/" + std::to_string(i) + ".bmp", 0);

        // stereo matching —— BM
        int mindisparity = 0;
        int ndisparities = 64;
        int SADWindowSize = 23;
        cv::Ptr <cv::StereoBM> bm = cv::StereoBM::create(ndisparities, SADWindowSize);
        bm->setBlockSize(SADWindowSize);
        bm->setMinDisparity(mindisparity);
        bm->setNumDisparities(ndisparities);
        bm->setPreFilterSize(7);
        bm->setPreFilterCap(31);
        bm->setSpeckleWindowSize(1000);   //1000
        bm->setSpeckleRange(20);
        bm->setTextureThreshold(100);
        bm->setUniquenessRatio(1);
        bm->setDisp12MaxDiff(1);
        Mat disp;
        //cv::copyMakeBorder(trans_cam1, trans_cam1, 8, 8, 8, 8, BORDER_REPLICATE);
        //cv::copyMakeBorder(trans_cam2, trans_cam2, 8, 8, 8, 8, BORDER_REPLICATE);
        bm->compute(hand, pattern, disp);

        disp.convertTo(disp, CV_32F, 1.0 / 16); //除以16得到真实视差值
        Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);
        cv::medianBlur(disp, disp, 5);
        normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);

        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
        Mat out;
        dilate(disp8U, disp8U, element);

        imwrite("/Users/ccfeng/Desktop/data/data/paper_exp/Fig.4/hand/noisy/depth_" + std::to_string(i) + ".bmp", disp8U);

        //save disparity to .txt
        ofstream outfile;
        outfile.open("/Users/ccfeng/Desktop/data/data/paper_exp/Fig.4/hand/noisy/disp_value_"+std::to_string(i)+".txt");
        for (int x = 0; x < disp8U.cols; x++){
            for (int y = 0; y < disp8U.rows; y++){
                float value = (float)disp8U.at<uchar>(y,x);
                outfile << value << "\n";
            }
        }
        outfile.close();
    }
    return 0;
}
