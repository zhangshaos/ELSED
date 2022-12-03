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
 * @return cv:Mat CV_8U
 */
cv::Mat createLineMapFromFile(const std::string& file, const cv::Mat &cMap=cv::Mat())
{
  auto colorImg = CV_Imread1920x1080(file);
  upm::ELSED lineDetector;
  //upm::Segments segs = lineDetector.detect(colorImg);
  upm::ImageEdges edges = lineDetector.detectEdges(colorImg);
  //auto colorMap1 = colorImg.clone(),
  //     colorMap2 = colorImg.clone();
  //drawSegments(colorMap1, segs, CV_RGB(0, 255, 0), 1);
  //drawEdges(colorMap2, edges, CV_RGB(0, 255, 0));
  //cv::imshow("Segs", colorMap1);
  //cv::imshow("Edges", colorMap2);
  //cv::waitKey();
  const int Rows = colorImg.size[0], Cols = colorImg.size[1];
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
      int deltaY = e.back().y - e.front().y,
          deltaX = e.back().x - e.front().x;
      //检测边e两侧的类型(cMap.ID)是否一致
      bool isTexture = true;
      for (const auto &px: e)
      {

      }
      if (!isTexture)
        for (const auto &px: e)
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
      // 计算当前像素resultCls.at<int32_t>(i, j)的类别。
      //
      // 两个像素联通，必须符合以下条件：
      // 0.L-inf距离为1
      // 1.对应cMap聚类图中同一个聚类中心
      // 2.被比较的那个像素不处于边缘图lineMap上
      std::set<int> connectedCls;//与当前像素联通的像素，他们的类别应该被合并
      int minConnectedCls = std::numeric_limits<int>::max();
      for (const Pxi32_t& yxOff :{Pxi32_t{0,-1},
                                  /*Pxi32_t{-1,-1},*/
                                  Pxi32_t{-1,0},
                                  /*Pxi32_t{-1,1}*/})
      {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (0 <= targetY && targetY < Rows && 0 <= targetX && targetX < Cols &&
            cMap.at<int32_t>(targetY, targetX) == cMap.at<int32_t>(i, j) &&
            lineMap.at<uchar>(targetY, targetX) <= 0)
        {
          int32_t C = resultCls.at<int32_t>(targetY, targetX);
          assert(C >= 0);
          connectedCls.emplace(C);
          if (C < minConnectedCls)
            minConnectedCls = C;
        }
      }
      if (minConnectedCls >= std::numeric_limits<int>::max())
      {
        //新类别
        assert(connectedCls.empty());
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
  for (int i=0; i<Rows; ++i)
    for (int j=0; j<Cols; ++j)
    {
      int32_t C = resultCls.at<int32_t>(i, j);
      resultCls.at<int32_t>(i, j) = (int)shrinkClass[C];
    }
  return nCls;
}



int main(int argc, char* argv[])
{
  auto grayImg = createLineMapFromFile("../images/1_scene.png");
  cv::imshow("Gray", grayImg);
  cv::waitKey();
  return 0;
}