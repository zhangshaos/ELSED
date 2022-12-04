// 给定一个法向的初步聚类图，使用ELSED方法检测到的线段和连通性检测，
// 将聚类图的簇进一步分割开来
// Created by zxm on 2022/12/3.
//

#include <map>
#include <set>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <ELSED.h>
#include "Npy2cvMat.h"
#include "Debug.h"


cv::Mat CV_Imread1920x1080(const std::string& file)
{
  cv::Mat img = cv::imread(file);
  cv::resize(img, img, cv::Size(1440, 1080), 0, 0, cv::INTER_NEAREST);
  cv::resize(img, img, cv::Size(960, 720), 0, 0, cv::INTER_NEAREST);
  //cv::resize(img, img, cv::Size(480, 360), 0, 0, cv::INTER_NEAREST);
  //cv::resize(img, img, cv::Size(256, 192), 0, 0, cv::INTER_NEAREST);
  return img;
}


inline void
drawSegments(cv::Mat img,
             upm::Segments segs,
             const cv::Scalar &color,
             int thickness = 1,
             int lineType = cv::LINE_AA,
             int shift = 0)
{
  for (const upm::Segment &seg: segs)
    cv::line(img, cv::Point2f(seg[0], seg[1]), cv::Point2f(seg[2], seg[3]), color, thickness, lineType, shift);
}


inline void
drawEdges(cv::Mat &img,
          const upm::ImageEdges &edges,
          const cv::Scalar &color)
{
  cv::Vec3b bgr((uchar)color[2], (uchar)color[1], (uchar)color[0]);
  for (const auto& e : edges)
    for (const auto& px : e)
      // OpenCV is BGR
      img.at<cv::Vec3b>((int)px.y, (int)px.x) = bgr;
}





/**
 * 主要函数2
 * 读取RGB图片file，运行ELSED算法检测所有线段。
 * 如果!cMap.empty，还会筛除掉那些纹理线：假设纹理线上所有像素的对应cMap.ID完全一致。
 * @param file
 * @param clustersMap CV_32S
 * @return cv:Mat CV_8U
 */
cv::Mat createLineMapFromFile(const std::string& file,
                              const cv::Mat &cMap=cv::Mat())
{
  assert(cMap.empty() || cMap.type()==CV_32S);
  auto colorImg = CV_Imread1920x1080(file);
  int Rows = colorImg.size[0], Cols = colorImg.size[1];
  assert(cMap.empty() || (Rows>=cMap.size[0] && Cols>=cMap.size[1]));
  upm::ELSED lineDetector;
  upm::ImageEdges edges = lineDetector.detectEdges(colorImg);
  if (!cMap.empty() && Rows > cMap.size[0] && Cols > cMap.size[1])
  {
    //线段检测结果放缩到cMap大小
    double scaleY = (double)cMap.size[0] / Rows,
           scaleX = (double)cMap.size[1] / Cols;
    Rows = cMap.size[0];
    Cols = cMap.size[1];
    for (auto& e : edges)
      for (auto& px : e)
      {
        px.y = int((px.y + 0.5) * scaleY);
        px.x = int((px.x + 0.5) * scaleX);
      }
  }
  //绘制edge热图result
  cv::Mat result(Rows, Cols, CV_8U, cv::Scalar_<uint8_t>(0));
  for (const auto& e : edges)
  {
    if (e.size() < 2)
      continue;
    if (cMap.empty())
      for (const auto &px: e)
        result.at<uint8_t>(px.y, px.x) = (uint8_t) -1;
    else
    {
      //计算每个像素应该向两侧偏移的距离(y0,x0)和(y1,x1)
      const float deltaY = float(e.back().y - e.front().y),
                  deltaX = float(e.back().x - e.front().x);
      const float Radius = sqrt(5.991f);//5.991 is our ORB_SLAM2 error threshold
      const float dy = (deltaY / sqrt(deltaY * deltaY + deltaX * deltaX)) * Radius,
                  dx = (deltaX / sqrt(deltaY * deltaY + deltaX * deltaX)) * Radius;
      const float y0 = -dx, x0 = dy, y1 = dx, x1 = -dy;//(dy,dx) 分别逆时针、顺时针转动90°
      //检测边e两侧的类型(cMap.ID)是否一致
      size_t nDiffSide = 0;
      for (const auto &px: e)
      {
        int iy0 = int(px.y + 0.5f + y0),
            ix0 = int(px.x + 0.5f + x0),
            iy1 = int(px.y + 0.5f + y1),
            ix1 = int(px.x + 0.5f + x1);
        if (0 <= iy0 && iy0 < Rows && 0 <= ix0 && ix0 < Cols &&
            0 <= iy1 && iy1 < Rows && 0 <= ix1 && ix1 < Cols &&
            cMap.at<int32_t>(iy0, ix0) != cMap.at<int32_t>(iy1, ix1))
          ++nDiffSide;
      }
      if (nDiffSide > size_t(0.2f * e.size()))
        //如果边缘e上有足够多的像素左右不一致，则表明e是一条结构线而不是纹理线
        for (const auto &px : e)
          result.at<uint8_t>(px.y, px.x) = (uint8_t) -1;
    }
  }
  return result;
}


struct SimpleNumberUnion
{
  //parent[i] 表示类别i的父类别
  std::vector<uint32_t> parent;

  bool containClass(uint32_t c)
  {
    return parent.size() > c && parent[c] != uint32_t(-1);
  }

  uint32_t findRootClass(uint32_t c)
  {
    if (containClass(c))
      while (c != parent[c])
        c = parent[c];
    else
      c = (uint32_t)-1;
    return c;
  }

  bool tryInsertClass(uint32_t c)
  {
    if (c >= parent.size())
      parent.resize(c + 1, uint32_t(-1));
    if (!containClass(c))
    {
      parent[c] = c;
      return true;
    } else
      // this ID is existed.
      return false;
  }

  void unionClass(uint32_t c1, uint32_t c2)
  {
    tryInsertClass(c1);
    tryInsertClass(c2);
    c1 = findRootClass(c1);
    c2 = findRootClass(c2);
    if (c1 <= c2)
      parent[c2] = c1;
    else
      parent[c1] = c2;
  }

  uint32_t shrink(std::vector<uint32_t>* outShrinkClass=nullptr)
  {
    //合并类别中间的空白。
    // 在所有类型合并结束后，每个类别的根类别可能出现[2,2,2,0,0,5,5,5]这种情况，
    // 将其收缩为 [0,0,0,1,1,2,2,2]
    std::vector<uint32_t> shrinkClass(parent.size(), (uint32_t)-1);
    uint32_t startShrinkCls = 0;
    std::map<uint32_t, uint32_t> rootClsToShrinkCls;
    for (uint32_t i = 0, iEnd = (uint32_t)parent.size(); i < iEnd; ++i)
    {
      if (!containClass(i))
        //并查集中没有类别i
        continue;
      uint32_t rootC = findRootClass(i);
      parent[i] = rootC;
      if (rootClsToShrinkCls.count(rootC))
        shrinkClass[i] = rootClsToShrinkCls.at(rootC);
      else
      {
        rootClsToShrinkCls[rootC] = startShrinkCls;
        shrinkClass[i] = startShrinkCls;
        ++startShrinkCls;
      }
    }
    {// 测试 shrink 是否正常
      for (uint32_t i = 0, iEnd = (uint32_t)parent.size(); i < iEnd; ++i)
      {
        assert(containClass(i) == (shrinkClass[i] != (uint32_t)-1));
        if (!containClass(i))
          continue;
        uint32_t oriC = findRootClass(i), newC = shrinkClass[i];
        assert(rootClsToShrinkCls.count(oriC));
        assert(rootClsToShrinkCls.at(oriC) == newC);
      }
    }
    if (outShrinkClass)
      outShrinkClass->swap(shrinkClass);
      // -1表示没有根类别
    return startShrinkCls;
  }
};


/**
 * 主要函数1
 * 给定一个法向的初步聚类图，使用ELSED方法检测到的线段和Two-Pass连通性检测，将聚类图的簇进一步分割开来
 * @param[in&out] cMap CV_32S
 * @param[in] lineMap CV_8U
 * @return 最终分割图的类别数量
 */
int segmentPlaneClusters(cv::Mat &cMap, const cv::Mat &lineMap)
{
  assert(cMap.type() == CV_32S);
  assert(lineMap.type() == CV_8U);
  assert(cMap.size == lineMap.size);
  using Pxi32_t = std::pair<int,int>;
  const int Rows = cMap.size[0], Cols = cMap.size[1];
  // first pass
  SimpleNumberUnion mergeCls;
  cv::Mat resultCls(Rows, Cols, CV_32S, -1);
  int nextCls = 0;
  for (int i=0; i<Rows; ++i)
    for (int j=0; j<Cols; ++j)
    {
      std::cout
      << "Cur: "
      << i
      << ','
      << j
      << ". ";
      // 计算当前像素resultCls.at<int32_t>(i, j)的类别。
      //
      // 两个像素联通，必须符合以下条件：
      // 0.L-inf距离为1
      // 1.对应cMap聚类图中同一个聚类中心
      // 2.被比较的那个像素不处于边缘图lineMap上
      std::set<int> connectedCls;//与当前像素联通的像素，他们的类别应该被合并
      int minConnectedCls = std::numeric_limits<int>::max();
      for (const Pxi32_t& yxOff :{Pxi32_t{0,-1},
                                  Pxi32_t{-1,-1},
                                  Pxi32_t{-1,0},
                                  Pxi32_t{-1,1}})
      {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (0 <= targetY && targetY < Rows &&
            0 <= targetX && targetX < Cols)
        {
          if (cMap.at<int32_t>(targetY, targetX) == cMap.at<int32_t>(i, j)) {
            if (true || lineMap.at<uint8_t>(targetY, targetX) <= 0) {
              int32_t C = resultCls.at<int32_t>(targetY, targetX);
              assert(C >= 0);
              connectedCls.emplace(C);
              if (C < minConnectedCls)
                minConnectedCls = C;
            }
          }
        }
      }
      std::cout
      << minConnectedCls
      << ' ';
      if (connectedCls.empty())
      {
        //新类别
        mergeCls.tryInsertClass(nextCls);
        resultCls.at<int32_t>(i, j) = nextCls++;
      }
      else
      {
        if (connectedCls.size() > 1)
          for(int C : connectedCls)
            if (C != minConnectedCls)
              mergeCls.unionClass(minConnectedCls, C);
        resultCls.at<int32_t>(i, j) = minConnectedCls;
      }
      std::cout
      << "class ID="
      << resultCls.at<int32_t>(i, j)
      << '\n';
    }
  std::vector<uint32_t> shrinkClass;
  int nCls = (int)mergeCls.shrink(&shrinkClass);
  // second pass
  for (int i=0; i<Rows; ++i)
    for (int j=0; j<Cols; ++j)
    {
      int32_t C = resultCls.at<int32_t>(i, j);
      resultCls.at<int32_t>(i, j) = (int)shrinkClass[C];
    }
  swap(cMap, resultCls);
  return nCls;
}


//PASS
void test()
{
  cv::Mat clustersMap(5, 5, CV_32S, -1);
  clustersMap.at<int32_t>(2,1) = 0;
  clustersMap.at<int32_t>(3,1) = 0;
  clustersMap.at<int32_t>(4,0) = 1;
  clustersMap.at<int32_t>(4,1) = 1;
  clustersMap.at<int32_t>(4,2) = 1;
  clustersMap.at<int32_t>(4,3) = 1;
  clustersMap.at<int32_t>(4,4) = 2;
  clustersMap.at<int32_t>(3,4) = 2;
  clustersMap.at<int32_t>(2,4) = 2;
  clustersMap.at<int32_t>(3,2) = 3;
  clustersMap.at<int32_t>(2,2) = 3;
  clustersMap.at<int32_t>(1,2) = 3;
  clustersMap.at<int32_t>(1,1) = 3;
  clustersMap.at<int32_t>(0,1) = 3;
  clustersMap.at<int32_t>(0,2) = 3;
  std::cout << clustersMap << std::endl;
  DrawClusters("../images/clusters0.png", clustersMap);
  cv::Mat grayImg(5, 5, CV_8U, cv::Scalar_<uint8_t>(0));
  grayImg.at<uint8_t>(0,0) = -1;
  grayImg.at<uint8_t>(1,0) = -1;
  grayImg.at<uint8_t>(2,0) = -1;
  grayImg.at<uint8_t>(3,0) = -1;
  grayImg.at<uint8_t>(0,2) = -1;
  grayImg.at<uint8_t>(1,2) = -1;
  grayImg.at<uint8_t>(2,2) = -1;
  grayImg.at<uint8_t>(3,2) = -1;
  grayImg.at<uint8_t>(3,3) = -1;
  grayImg.at<uint8_t>(3,4) = -1;
  std::cout << grayImg << std::endl;
  auto N = segmentPlaneClusters(clustersMap, grayImg);
  std::cout << "Number of Classes: " << N << '\n';
  DrawClusters("../images/clusters1.png", clustersMap);
}


cv::Mat cv32F3To32FC3(const cv::Mat& m, const cv::Mat& mask)
{
  assert(m.type() == CV_32F);
  assert(mask.type() == CV_8U);
  cv::Mat o(m.size[0], m.size[1], CV_32FC3);
  for (int i=0; i<m.size[0]; ++i)
    for (int j=0; j<m.size[1]; ++j)
    {
      if (mask.at<uint8_t>(i, j) <= 1)
      {
        o.at<cv::Vec3f>(i, j) = {1, 0, 0};//指向照片内部
      } else
      {
        o.at<cv::Vec3f>(i, j)[0] = m.at<float>(i, j, 0);
        o.at<cv::Vec3f>(i, j)[1] = m.at<float>(i, j, 1);
        o.at<cv::Vec3f>(i, j)[2] = m.at<float>(i, j, 2);
      }
    }
  return o;
}


//法向量聚类方法
void testNormalClustering()
{
  cv::Mat normalMap = cvDNN::blobFromNPY(R"(E:\VS_Projects\MonoPlanner\example\data\55_normal.npy)",
                                         CV_32F);
  cv::Mat mask = cvDNN::blobFromNPY(R"(E:\VS_Projects\MonoPlanner\example\data\55_mask.npy)",
                                    CV_8U);
  normalMap = cv32F3To32FC3(normalMap, mask);
  DrawNormals("../images/normals.png", normalMap);
  using Pxi32_t = std::pair<int,int>;
  const int Rows = normalMap.size[0], Cols = normalMap.size[1];
  auto isValidIndex = [&Rows,&Cols](int y, int x)->bool
  {
    return 0 <= y && y < Rows && 0 <= x && x < Cols;
  };
  //法向归一化
  for (int i=0; i<Rows; ++i)
    for (int j=0; j<Cols; ++j)
    {
      const auto& v3 = normalMap.at<cv::Vec3f>(i, j);
      normalMap.at<cv::Vec3f>(i, j) = cv::normalize(v3);
    }
  // first pass
  SimpleNumberUnion mergeCls;
  cv::Mat resultCls(Rows, Cols, CV_32S, -1);
  int nextCls = 0;
  for (int i=0; i<Rows; ++i)
    for (int j=0; j<Cols; ++j)
    {
      // 计算当前像素resultCls.at<int32_t>(i, j)的类别。
      //
      // 两个像素联通，必须符合以下条件：
      // 0.L-inf距离为1
      // 1.对应cMap聚类图中同一个聚类中心
      // 2.被比较的那个像素不处于边缘图lineMap上
      std::set<int> connectedCls;//与当前像素联通的像素，他们的类别应该被合并
      int minConnectedCls = std::numeric_limits<int>::max();
      for (const Pxi32_t& yxOff :{Pxi32_t{0,-1},
                                  Pxi32_t{-1,-1},
                                  Pxi32_t{-1,0},
                                  Pxi32_t{-1,1}})
      {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (isValidIndex(targetY, targetX))
        {
          //计算targetYX和ij的曲率
          bool isContinued = true;
          {
            const cv::Vec3f v0 = isValidIndex(targetY + yxOff.first, targetX + yxOff.second) ?
              normalMap.at<cv::Vec3f>(targetY + yxOff.first, targetX + yxOff.second) :
              normalMap.at<cv::Vec3f>(targetY, targetX);
            const cv::Vec3f v1 = normalMap.at<cv::Vec3f>(targetY, targetX);
            const cv::Vec3f v2 = normalMap.at<cv::Vec3f>(i, j);
            const cv::Vec3f v3 = isValidIndex(i - yxOff.first, j - yxOff.second) ?
              normalMap.at<cv::Vec3f>(i - yxOff.first, j - yxOff.second) :
              normalMap.at<cv::Vec3f>(i, j);
//            const double cos0 = v0.dot(v1), cos1 = v1.dot(v2), cos2 = v2.dot(v3);
//            const double ang0 = std::max(acos(cos0), CV_PI/180),
//                         ang1 = std::max(acos(cos1), CV_PI/180),
//                         ang2 = std::max(acos(cos2), CV_PI/180);
//            const double r0 = ang0 / ang1, r1 = ang1 / ang2;//r0,r1都在'1'附近
//            isContinued = abs(r0 - r1) < std::max(r0, r1) * 0.1;

            const cv::Vec3f v1Mean = cv::normalize(v0 + v1);
            const cv::Vec3f v2Mean = cv::normalize(v2 + v3);
            isContinued = acos(v1Mean.dot(v2Mean)) < 10 * CV_PI / 180;

//            isContinued = acos(v0.dot(v3)) < 10 * CV_PI / 180;
          }
          if (isContinued)
          {
            int32_t C = resultCls.at<int32_t>(targetY, targetX);
            assert(C >= 0);
            connectedCls.emplace(C);
            if (C < minConnectedCls)
              minConnectedCls = C;
          }
        }
      }//检测(i,j)与周围像素的连通性
      if (connectedCls.empty())
      {
        //新类别
        mergeCls.tryInsertClass(nextCls);
        resultCls.at<int32_t>(i, j) = nextCls++;
      }
      else
      {
        if (connectedCls.size() > 1)
          for(int C : connectedCls)
            if (C != minConnectedCls)
              mergeCls.unionClass(minConnectedCls, C);
        resultCls.at<int32_t>(i, j) = minConnectedCls;
      }
    }
  std::vector<uint32_t> shrinkClass;
  int nCls = (int)mergeCls.shrink(&shrinkClass);
  // second pass
  std::map<uint32_t, size_t> classCounter;//记录每个类别的元素（像素）个数
  for (int i=0; i<Rows; ++i)
    for (int j=0; j<Cols; ++j)
    {
      int32_t C = resultCls.at<int32_t>(i, j);
      C = (int32_t)shrinkClass[C];
      resultCls.at<int32_t>(i, j) = C;
      if (classCounter.count(C))
        ++classCounter.at(C);
      else
        classCounter[C] = 1;
    }
  std::cout << "nClass = " << nCls << '\n';
  // resultCls...
  DrawClusters("../images/normalClusters.png", resultCls);
  for (const auto& [c, n] : classCounter)
  {
    std::cout
    << "Class="
    << c
    << ", Count="
    << n
    << '\n';
  }
  //todo：需要对相邻的像素点（两者对应类别的计数在一个数量级之内）做合并
}


int main(int argc, char* argv[])
{
  try {
    testNormalClustering();
//    cv::Mat clustersMap = cvDNN::blobFromNPY(R"(E:\VS_Projects\MonoPlanner\example\data\1_clusters.npy)",
//                                             CV_32S);
//    DrawClusters("../images/clusters0.png", clustersMap);
//    std::cout << "Number of original Clusters: " << max<int32_t>(clustersMap) << '\n';
//    auto grayImg = createLineMapFromFile("../images/1_scene.png", clustersMap);
//    auto N = segmentPlaneClusters(clustersMap, grayImg);
//    std::cout << "Number of Classes: " << N << '\n';
//    DrawClusters("../images/clusters1.png", clustersMap);
//    //cv::imwrite("../images/out1.png", grayImg);
//    //grayImg = createLineMapFromFile("../images/1_scene.png");
//    //cv::imwrite("../images/out0.png", grayImg);
//    //cv::imshow("Gray", grayImg);
//    //cv::waitKey();
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
  } catch (...)
  {
    std::cout << "Unknown Error!" << std::endl;
  }
  return 0;
}