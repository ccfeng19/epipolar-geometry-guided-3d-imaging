#include <iostream>
#include <opencv2/opencv.hpp>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace cv;
using namespace std;

int main() {
    //projector 1280*720
    //basler camera 1920*1200
    //flir camera 2048*1536
    //相机在左 投影仪在右

    //1.读入标定参数
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

    //2.投影仪散斑图映射
    Mat I_proj=imread("/Users/ccfeng/Desktop/trans/rand_peri_img.png",0); //读入单通道灰度图，8位
    if (!I_proj.data){
        cout<<"load image error!";
        return -1;
    }
    int cols=I_proj.cols;
    int rows=I_proj.rows;

    Mat disparityToDepthQ_p(3,3,CV_64FC1,cv::Scalar(0));
    cv::stereoRectify(cameraIntrinsicMatrix,cameraDistortionMatrix,projectorIntrinsicMatrix,projectorDistortionMatrix,
                      cv::Size(cols,rows),extrinsicRotation,extrinsicTranslation,cameraRectifiedRotation,projectorRectifiedRotation,
                      cameraRectifiedProjectMatrix,projectorRectifiedProjectMatrix,disparityToDepthQ_p,CALIB_ZERO_DISPARITY,-1);
    cv::Mat map_p1,map_p2;
    cv::initUndistortRectifyMap(projectorIntrinsicMatrix, projectorDistortionMatrix, projectorRectifiedRotation, projectorRectifiedProjectMatrix,
                                cv::Size(cols, rows), CV_32FC1, map_p1, map_p2);

    Mat trans_I(rows,cols,CV_8U,cv::Scalar::all(0));
    //x代表列值，y代表行值
    float x,y;
    int x_l,y_l,x_h,y_h;
    for (int i = 0; i < rows; i++) {
        auto *m1f = map_p1.ptr<float>(i);
        auto *m2f = map_p2.ptr<float>(i);
        for (int j = 0; j < cols; j++) {
            x = m1f[j];
            y = m2f[j];
            x_l = floor(x);
            y_l = floor(y);
            x_h = ceil(x);
            y_h = ceil(y);
            if (x_l >= 0 && x_l < cols && y_l >= 0 && y_l < rows)
                trans_I.at<uchar>(y_l,x_l) += I_proj.at<uchar>(i,j)*(float(x_h) - x) * (float(y_h) - y);
            if (x_l >= 0 && x_l < cols && y_h >= 0 && y_h < rows)
                trans_I.at<uchar>(y_h,x_l) += I_proj.at<uchar>(i,j)*(float(x_h) - x) * (y - float(y_l));
            if (x_h >= 0 && x_h < cols && y_l >= 0 && y_l < rows)
                trans_I.at<uchar>(y_l,x_h) += I_proj.at<uchar>(i,j)*(x - float(x_l)) * (float(y_h) - y);
            if (x_h >= 0 && x_h < cols && y_h >= 0 && y_h < rows)
                trans_I.at<uchar>(y_h,x_h) += I_proj.at<uchar>(i,j)*(x - float(x_l)) * (y - float(y_l));
        }
    }

    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            if (trans_I.at<uchar>(i,j)<12.75)
                trans_I.at<uchar>(i,j)=0;
            else
                trans_I.at<uchar>(i,j)=255;

        }
    }
    cv::medianBlur(trans_I,trans_I,5);
    imwrite("/Users/ccfeng/Desktop/trans/20201128.png",trans_I);

    //将projector_transformed通过remap映射回来 观察是否映射正确 调试用
    Mat remap_trans_I;
    cv::remap(trans_I,remap_trans_I,map_p1,map_p2,cv::INTER_LINEAR,cv::BORDER_CONSTANT,0);
    imwrite("/Users/ccfeng/Desktop/trans/remap_projector.png",remap_trans_I);

    return 0;

}