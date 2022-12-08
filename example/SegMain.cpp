// 给定一个法向的初步聚类图，使用ELSED方法检测到的线段和连通性检测，
// 将聚类图的簇进一步分割开来
// Created by zxm on 2022/12/3.
//

#include <map>
#include <set>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo.hpp>

#include "ELSED.h"
#include "Npy2CVMat.h"
#include "Tools.h"
#include "PlaneDetector.h"


//PASS
void testSegmentByLines() {
  cv::Mat clustersMap(5, 5, CV_32S, -1);
  clustersMap.at<int32_t>(2, 1) = 0;
  clustersMap.at<int32_t>(3, 1) = 0;
  clustersMap.at<int32_t>(4, 0) = 1;
  clustersMap.at<int32_t>(4, 1) = 1;
  clustersMap.at<int32_t>(4, 2) = 1;
  clustersMap.at<int32_t>(4, 3) = 1;
  clustersMap.at<int32_t>(4, 4) = 2;
  clustersMap.at<int32_t>(3, 4) = 2;
  clustersMap.at<int32_t>(2, 4) = 2;
  clustersMap.at<int32_t>(3, 2) = 3;
  clustersMap.at<int32_t>(2, 2) = 3;
  clustersMap.at<int32_t>(1, 2) = 3;
  clustersMap.at<int32_t>(1, 1) = 3;
  clustersMap.at<int32_t>(0, 1) = 3;
  clustersMap.at<int32_t>(0, 2) = 3;
  std::cout << clustersMap << std::endl;
  zxm::DrawClusters("../dbg/testSegmentByLines0.png", clustersMap);
  cv::Mat lineMap(5, 5, CV_8U, cv::Scalar_<uint8_t>(0));
  lineMap.at<uint8_t>(0, 0) = -1;
  lineMap.at<uint8_t>(1, 0) = -1;
  lineMap.at<uint8_t>(2, 0) = -1;
  lineMap.at<uint8_t>(3, 0) = -1;
  lineMap.at<uint8_t>(0, 2) = -1;
  lineMap.at<uint8_t>(1, 2) = -1;
  lineMap.at<uint8_t>(2, 2) = -1;
  lineMap.at<uint8_t>(3, 2) = -1;
  lineMap.at<uint8_t>(3, 3) = -1;
  lineMap.at<uint8_t>(3, 4) = -1;
  std::cout << lineMap << std::endl;
  int N = zxm::SegmentByLines(clustersMap, lineMap);
  std::cout << "Number of Classes: " << N << '\n';
  zxm::DrawClusters("../dbg/testSegmentByLines1.png", clustersMap);
}


//PASS法向量聚类方法
void testDetectPlanes() {
  using Pxi32_t = std::pair<int, int>;
  cv::Mat colorImg =
    zxm::CV_Imread1920x1440(R"(../images/99_scene.png)", cv::IMREAD_COLOR, cv::INTER_NEAREST);
  cv::Mat mask =
    cvDNN::blobFromNPY(R"(E:\VS_Projects\MonoPlanner\example\data\99_mask.npy)", CV_8U);
  cv::Mat normalMap =
    cvDNN::blobFromNPY(R"(E:\VS_Projects\MonoPlanner\example\data\99_normal.npy)", CV_32F);
  normalMap = zxm::CV_Convert32FTo32FC3(normalMap, mask);
  zxm::DrawNormals("../dbg/inNormals.png", normalMap);
  //
  cv::Mat planesMat = zxm::DetectPlanes(colorImg, normalMap, true);
  //
  std::cout << "nClass = " << zxm::cvMax<int32_t>(planesMat) + 1 << '\n';
  zxm::DrawClusters("../dbg/outPlanesMap.png", planesMat);
}


int main(int argc, char *argv[]) {
  try {
    //PASS
    //testSegmentByLines();
    //
    testDetectPlanes();
    zxm::CheckMathError();
    //
    //testRaster();
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
  } catch (...) {
    std::cout << "Unknown Error!" << std::endl;
  }
  return 0;
}