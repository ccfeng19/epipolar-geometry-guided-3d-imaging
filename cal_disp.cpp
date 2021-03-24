#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace cv;
using namespace std;


int main(){

    // 读入标定参数
    Mat projectorIntrinsicMatrix(3,3,CV_64FC1,cv::Scalar(0));
    Mat projectorDistortionMatrix(1,5,CV_64FC1,cv::Scalar(0));
    Mat cameraIntrinsicMatrix(3,3,CV_64FC1,cv::Scalar(0));
    Mat cameraDistortionMatrix(1,5,CV_64FC1,cv::Scalar(0));
    Mat extrinsicRotation(3,3,CV_64FC1,cv::Scalar(0));
    Mat extrinsicTranslation(3,1,CV_64FC1,cv::Scalar(0));
    Mat projectorRectifiedRotation(3,3,CV_64FC1,cv::Scalar(0));
    Mat projectorRectifiedProjectMatrix(3,4,CV_64FC1,cv::Scalar(0));
    Mat cameraRectifiedRotation(3,3,CV_64FC1,cv::Scalar(0));
    Mat cameraRectifiedProjectMatrix(3,4,CV_64FC1,cv::Scalar(0));

    cv::FileStorage file_cam("/Users/ccfeng/Desktop/trans/20201128/camera.yml", cv::FileStorage::READ);
    if (!file_cam.isOpened()) {
        std::cerr << "Calibration: read camera parameters failed!" << endl;
        return -1;
    }
    file_cam["cameraIntrinsicParameters"] >> cameraIntrinsicMatrix;
    file_cam["cameraDistortionParameters"] >> cameraDistortionMatrix;
    file_cam.release();
    cv::FileStorage file_proj("/Users/ccfeng/Desktop/trans/20201128/projector.yml", cv::FileStorage::READ);
    if (!file_proj.isOpened()) {
        std::cerr << "Calibration: read camera parameters failed!" << endl;
        return -1;
    }
    file_proj["projectorIntrinsicParameters"] >> projectorIntrinsicMatrix;
    file_proj["projectorDistortionParameters"] >> projectorDistortionMatrix;
    file_proj.release();
    cv::FileStorage file_extrin("/Users/ccfeng/Desktop/trans/20201128/extrinsic.yml", cv::FileStorage::READ);
    if (!file_extrin.isOpened()) {
        std::cerr << "Calibration: read camera parameters failed!" << endl;
        return -1;
    }
    file_extrin["projectorRotationParameters"] >> extrinsicRotation;;
    file_extrin["projectorTranslationParameters"] >> extrinsicTranslation;
    file_extrin.release();

    // 读入相机视野两张图片并分别remap I_cam1代表相机视野 I_cam2代表投影视野
    for (int sigma=0;sigma<=7;sigma+=1) {
        Mat I_cam1 = imread("/Users/ccfeng/Desktop/cal_disp/7/" + std::to_string(sigma) + ".bmp", 0);
        int cam_cols = I_cam1.cols;
        int cam_rows = I_cam1.rows;
        Mat disparityToDepthQ_c(3, 3, CV_64FC1, cv::Scalar(0));
        cv::stereoRectify(cameraIntrinsicMatrix, cameraDistortionMatrix, projectorIntrinsicMatrix,
                          projectorDistortionMatrix,
                          cv::Size(cam_cols, cam_rows), extrinsicRotation, extrinsicTranslation,
                          cameraRectifiedRotation, projectorRectifiedRotation,
                          cameraRectifiedProjectMatrix, projectorRectifiedProjectMatrix, disparityToDepthQ_c,
                          CALIB_ZERO_DISPARITY, -1);
        cv::Mat map_c1, map_c2;
        cv::initUndistortRectifyMap(cameraIntrinsicMatrix, cameraDistortionMatrix, cameraRectifiedRotation,
                                    cameraRectifiedProjectMatrix,
                                    cv::Size(cam_cols, cam_rows), CV_32FC1, map_c1, map_c2);

        Mat trans_cam1, trans_cam2;
        cv::remap(I_cam1, trans_cam1, map_c1, map_c2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

        imwrite("/Users/ccfeng/Desktop/cal_disp/7/remap_" + std::to_string(sigma) + ".bmp", trans_cam1);

    }
        return 0;
}

