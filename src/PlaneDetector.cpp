//
// Created by zxm on 2022/12/5.
//


#include <map>
#include <vector>
#include <opencv2/core.hpp>

#include "ELSED.h"
#include "../HyperConfig.h"
#include "PlaneDetector.h"
#include "Tools.h"


cv::Mat zxm::DetectPlanes(const cv::Mat &colorImg,
                          const cv::Mat &normalMap,
                          bool enableDebug) {
  cv::Mat clusterMap;
  zxm::ClusteringByNormal(clusterMap, normalMap, true);
  cv::Mat clusterColorMap;
  if (enableDebug)
    clusterColorMap = DrawClusters("../dbg/NormalClusters.png", clusterMap);
  auto lineMap = zxm::CreateStructureLinesMap(colorImg, normalMap);
  if (enableDebug)
    DrawClusters("../dbg/Lines.png", lineMap);
  if (enableDebug) {
    const int Rows = clusterColorMap.size[0], Cols = clusterColorMap.size[1];
    CV_Assert(Rows == lineMap.size[0] && Cols == lineMap.size[1]);
    for (int i = 0; i < Rows; ++i)
      for (int j = 0; j < Cols; ++j)
        if (lineMap.at<int32_t>(i, j) > 0)
          clusterColorMap.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
    CV_ImWriteWithPath("../dbg/BlendNormalClustersAndLines.png", clusterColorMap);
  }
  zxm::SegmentByLines(clusterMap, lineMap, enableDebug);
  return clusterMap;
}


int zxm::ClusteringByNormal(cv::Mat &clusterMap,
                            const cv::Mat &normalMap,
                            bool enableDebug) {
  using Pxi32_t = std::pair<int, int>;
  CV_Assert(normalMap.type() == CV_32FC3);

  const int Rows = normalMap.size[0], Cols = normalMap.size[1];
  auto isValidIndex = [&Rows, &Cols](int y, int x) -> bool {
    return 0 <= y && y < Rows && 0 <= x && x < Cols;
  };
  auto isContinuedAngle = [](const cv::Vec3f &v0, const cv::Vec3f &v1) -> bool {
    return acos(clamp(v0.dot(v1), -1.f, 1.f)) <= (TH_CONTINUED_ANGLE * CV_PI / 180);
  };
  auto isDiscreted = [](const cv::Vec3f &v0, const cv::Vec3f &v1) -> bool {
    return acos(clamp(v0.dot(v1), -1.f, 1.f)) >= (2 * TH_CONTINUED_ANGLE * CV_PI / 180);
  };
  // first pass
  ClassUnion mergeCls;
  cv::Mat resultCls(Rows, Cols, CV_32S, -1);
  int nextCls = 0;
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      const cv::Vec3f &v2 = normalMap.at<cv::Vec3f>(i, j);
      std::map<int, cv::Vec3f> connectedCls;//与当前像素联通的像素，他们的类别应该被合并
      int minConnectedCls = std::numeric_limits<int>::max();
      for (const Pxi32_t &yxOff: {Pxi32_t{0, -1},
                                  Pxi32_t{-1, -1},
                                  Pxi32_t{-1, 0},
                                  Pxi32_t{-1, 1}}) {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (isValidIndex(targetY, targetX)) {
          bool isContinued = false;
          const cv::Vec3f &v1 = normalMap.at<cv::Vec3f>(targetY, targetX);
          /*
          const cv::Vec3f v0 =
            isValidIndex(targetY+yxOff.first, targetX+yxOff.second) && !isOnLine(targetY,targetX) ?
            normalMap.at<cv::Vec3f>(targetY + yxOff.first, targetX + yxOff.second) : v1;
          */
          /*const cv::Vec3f &v2 = normalMap.at<cv::Vec3f>(i, j);*/
          const cv::Vec3f v3 = isValidIndex(i - yxOff.first, j - yxOff.second) ?
                               normalMap.at<cv::Vec3f>(i - yxOff.first, j - yxOff.second) : v2;
          //
          isContinued = (isContinuedAngle(v1, v2) &&
                         isContinuedAngle(v2, v3) &&
                         isContinuedAngle(v1, v3)) ||//保证圆润的角依旧可以区分
                        (isContinuedAngle(v1, v2) &&
                         isDiscreted(v2, v3));//找补一些邻近的正面边缘的像素
          zxm::CheckMathError();
          if (isContinued) {
            int32_t C = resultCls.at<int32_t>(targetY, targetX);
            if (C < 0)
              throw std::logic_error("Processed label of (targetY,targetX) should >= 0!"
                                     " in zxm::ClusteringByNormal()");
            connectedCls.emplace(C, v1);
            if (C < minConnectedCls)
              minConnectedCls = C;
          }
        }
      }//检测(i,j)与周围像素的连通性
      if (connectedCls.empty()) {
        //新类别
        mergeCls.tryInsertClass(nextCls);
        resultCls.at<int32_t>(i, j) = nextCls++;
      } else {
        if (connectedCls.size() > 1) {
          const auto &normal0 = connectedCls.at(minConnectedCls);
          for (const auto &[C, normal1] : connectedCls) {
            if (C != minConnectedCls && isContinuedAngle(normal0, normal1))
              //法向量的连通性很难传递！
              mergeCls.unionClass(minConnectedCls, C);
          }
        }
        resultCls.at<int32_t>(i, j) = minConnectedCls;
      }
    }//one pass over.
  }

  if (enableDebug)
    zxm::DrawClusters("../dbg/RawNormalClusters.png", resultCls);

  std::vector<uint32_t> shrinkClass;
  int nCls = (int) mergeCls.shrink(&shrinkClass);
  // second pass
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      int32_t C = resultCls.at<int32_t>(i, j);
      C = (int32_t) shrinkClass[C];
      resultCls.at<int32_t>(i, j) = C;
    }
  }
  // resultCls...
  swap(resultCls, clusterMap);
  return nCls;
}


namespace zxm {
  // 当斜率0<k<1是的中值画线法(ix0,iy0)->(ix1,iy1)
  std::vector<upm::Pixel>
  raster_core(int iy0, int ix0, int iy1, int ix1) {
    std::vector<upm::Pixel> result;
    double k = double(iy1 - iy0) / (ix1 - ix0);
    double b = (iy1 + 0.5) - k * (ix1 + 0.5);
    result.emplace_back(ix0, iy0);
    int y = iy0;
    for (int x = ix0; x < ix1; ++x) {
      double midX = x + 1.5, midY = y + 1.0;
      double d = k * midX + b - midY;
      if (d >= 0)
        ++y;
      result.emplace_back(x + 1, y);
    }//over
    return result;
  }

  std::vector<upm::Pixel>
  raster(int iy0, int ix0, int iy1, int ix1) {
    using namespace std;
    if (ix0 > ix1) {
      //保证(ix0,iy0)是在x-y坐标系更靠左侧
      swap(ix0, ix1);
      swap(iy0, iy1);
    }
    std::vector<upm::Pixel> result;
    if (iy1 == iy0) {
      //横线
      for (int i = ix0; i <= ix1; ++i)
        result.emplace_back(i, iy0);
    } else if (ix1 == ix0) {
      //纵线
      if (iy0 > iy1)
        swap(iy0, iy1);//保证iy0在底端
      for (int i = iy0; i <= iy1; ++i)
        result.emplace_back(ix0, i);
    } else {
      if (iy1 - iy0 == ix1 - ix0) {
        // k==1
        for (int i = ix0; i <= ix1; ++i)
          result.emplace_back(i, iy0 + (i - ix0));
      } else if (iy1 - iy0 == ix0 - ix1) {
        // k==-1
        for (int i = ix0; i <= ix1; ++i)
          result.emplace_back(i, iy0 - (i - ix0));
      } else {
        //只处理直线斜率k属于(0,1)之间的情况
        double k = double(iy1 - iy0) / (ix1 - ix0);
        if (0 < k && k < 1)
          result = raster_core(iy0, ix0, iy1, ix1);
        else if (k > 1) {
          //交换xy轴
          result = raster_core(ix0, iy0, ix1, iy1);
          std::for_each(std::execution::seq,
                        result.begin(),
                        result.end(),
                        [](upm::Pixel &px) {
                          swap(px.x, px.y);
                        });
        } else if (k > -1 && k < 0) {
          //x轴翻转，y -> -y
          result = raster_core(-iy0, ix0, -iy1, ix1);
          std::for_each(std::execution::seq,
                        result.begin(),
                        result.end(),
                        [](upm::Pixel &px) {
                          px.y *= -1;
                        });
        } else if (k < -1) {
          //x轴翻转，然后交换xy轴
          result = raster_core(ix0, -iy0, ix1, -iy1);
          std::for_each(std::execution::seq,
                        result.begin(),
                        result.end(),
                        [](upm::Pixel &px) {
                          swap(px.x, px.y);
                          px.y *= -1;
                        });
        } else
          throw std::logic_error("zxm::raster() Impossible!");
      }
    }
    if (result.size() > 1)
      return result;
    else
      return {};
  }
}


// 测试zxm::raster()算法
static void TestRaster() {
  auto l1 = zxm::raster(17, 121, 1, 1),
    l2 = zxm::raster(120, 77, 23, 23),
    l3 = zxm::raster(100, 1, 97, 33),
    l4 = zxm::raster(97, 11, 37, 29),
    l5 = zxm::raster(100, 34, 100, 100),
    l6 = zxm::raster(77, 120, 11, 120);
  std::cout << l1.front().y << ' ' << l1.front().x << ' ' << l1.back().y << ' ' << l1.back().x
            << " Should be " << "1 1 17 121" << '\n';
  std::cout << l2.front().y << ' ' << l2.front().x << ' ' << l2.back().y << ' ' << l2.back().x
            << " Should be " << "23 23 120 77" << '\n';
  std::cout << l3.front().y << ' ' << l3.front().x << ' ' << l3.back().y << ' ' << l3.back().x
            << " Should be " << "100 1 97 33" << '\n';
  std::cout << l4.front().y << ' ' << l4.front().x << ' ' << l4.back().y << ' ' << l4.back().x
            << " Should be " << "97 11 37 29" << '\n';
  std::cout << l5.front().y << ' ' << l5.front().x << ' ' << l5.back().y << ' ' << l5.back().x
            << " Should be " << "100 34 100 100" << '\n';
  std::cout << l6.front().y << ' ' << l6.front().x << ' ' << l6.back().y << ' ' << l6.back().x
            << " Should be " << "11 120 77 120" << '\n';
  std::cout << std::endl;
  auto vs = l1;
  vs.insert(vs.end(), l2.begin(), l2.end());
  vs.insert(vs.end(), l3.begin(), l3.end());
  vs.insert(vs.end(), l4.begin(), l4.end());
  vs.insert(vs.end(), l5.begin(), l5.end());
  vs.insert(vs.end(), l6.begin(), l6.end());
  cv::Mat result(192, 256, CV_8U, cv::Scalar_<uint8_t>(0));
  for (const auto v: vs) {
    result.at<uint8_t>(v.y, v.x) = 0xff;
  }
  zxm::CV_ImWriteWithPath("../dbg/raster.png", result);
}


cv::Mat zxm::CreateStructureLinesMap(const cv::Mat &colorImg,
                                     const cv::Mat &normalMap) {
  CV_Assert(colorImg.type() == CV_8UC3);
  CV_Assert(normalMap.type() == CV_32FC3);
  CV_Assert(colorImg.size[0] >= normalMap.size[0]);
  CV_Assert(colorImg.size[1] >= normalMap.size[1]);

  int Rows = colorImg.size[0], Cols = colorImg.size[1];
  upm::ELSED lineDetector;
  auto edges = lineDetector.detectEdges(colorImg);

  if (Rows > normalMap.size[0] && Cols > normalMap.size[1]) {
    //线段检测结果放缩到normalMap大小
    const float
      scaleY = (float) normalMap.size[0] / Rows,
      scaleX = (float) normalMap.size[1] / Cols;
    Rows = normalMap.size[0];
    Cols = normalMap.size[1];
    for (auto &e: edges) {
      for (auto &px: e) {
        px.y = int((px.y + 0.5) * scaleY);
        px.x = int((px.x + 0.5) * scaleX);
      }
    }
  }
  //判断线段是纹理线还是结构线（两侧深度or深度不一致）：用法向量代理深度检测结构线会有一点问题。
  auto isDiffSide = [&normalMap](int iy0, int ix0, int iy1, int ix1) {
    const auto
      &v0 = normalMap.at<cv::Vec3f>(iy0, ix0),
      &v1 = normalMap.at<cv::Vec3f>(iy1, ix1);
    const float angle = acos(clamp(v0.dot(v1), -1.f, 1.f));
    zxm::CheckMathError();
    return angle >= float(TH_DIFF_SIDE_ANGLE * CV_PI / 180);
  };
  //绘制边缘热图result
  cv::Mat result(Rows, Cols, CV_32S, cv::Scalar_<int32_t>(-1));
  int32_t startID = 1;
  for (const auto &e: edges) {
    if (e.size() < 2)
      continue;
    //计算每个像素应该向两侧偏移的距离(y0,x0)和(y1,x1)
    const float
      deltaY = float(e.back().y - e.front().y),
      deltaX = float(e.back().x - e.front().x);
    const float
      dy = (deltaY / sqrt(deltaY * deltaY + deltaX * deltaX)) * GAP_HALF_DIFF_SIDE,
      dx = (deltaX / sqrt(deltaY * deltaY + deltaX * deltaX)) * GAP_HALF_DIFF_SIDE;
    zxm::CheckMathError();
    const float
      y0 = -dx, x0 = dy,
      y1 = dx, x1 = -dy;//(dy,dx) 分别逆时针、顺时针转动90°
    //检测边e两侧的法向量是否一致
    size_t nDiffSide = 0;
    for (const auto &px: e) {
      int iy0 = int(px.y + 0.5f + y0),
        ix0 = int(px.x + 0.5f + x0),
        iy1 = int(px.y + 0.5f + y1),
        ix1 = int(px.x + 0.5f + x1);
      if (0 <= iy0 && iy0 < Rows && 0 <= ix0 && ix0 < Cols &&
          0 <= iy1 && iy1 < Rows && 0 <= ix1 && ix1 < Cols &&
          isDiffSide(iy0, ix0, iy1, ix1))
        ++nDiffSide;
    }
    if (nDiffSide >= std::max(size_t(STRUCTURE_LINE_RATIO * e.size()), size_t(1))) {
      //如果边缘e上有足够多的像素左右不一致，则表明e是一条结构线而不是纹理线
      for (const auto &px : e)
        result.at<int32_t>(px.y, px.x) = startID;
      ++startID;
    }
  }
  return result;
}


int zxm::SegmentByLines(cv::Mat &clusterMap,
                        const cv::Mat &lineMap,
                        bool enableDebug) {
  using Pxi32_t = std::pair<int, int>;
  CV_Assert(clusterMap.type() == CV_32S);
  CV_Assert(lineMap.type() == CV_32S);
  CV_Assert(clusterMap.size == lineMap.size);

  const int Rows = clusterMap.size[0], Cols = clusterMap.size[1];
  // first pass
  ClassUnion mergeCls;
  cv::Mat resultCls(Rows, Cols, CV_32S, -1);
  int nextCls = 0;
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      int32_t cCur = clusterMap.at<int32_t>(i, j);
      int32_t curLine = lineMap.at<int32_t>(i, j);
      if (curLine > 0)
        //线段上像素点的归属稍后再判断
        continue;
      //计算当前像素(i,j)和周围4领域像素的连通性
      std::set<int> connectedCls;//与当前像素联通的像素，他们的类别应该被合并
      int minConnectedCls = std::numeric_limits<int>::max();
      for (const Pxi32_t &yxOff: {Pxi32_t{0, -1},
                                  Pxi32_t{-1, 0}}) {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (0 <= targetY && targetY < Rows &&
            0 <= targetX && targetX < Cols) {
          int32_t cTarget = clusterMap.at<int32_t>(targetY, targetX);
          int32_t targetLine = lineMap.at<int32_t>(targetY, targetX);
          //解决穿透效应，对在线上的像素单独处理。
          //1、c和t都不在线上： 若类ID相等，则联通，并更新connectedCls和maxConnectedCls
          //2、c不在线上，t在： 不联通
          if (targetLine <= 0 && (cTarget == cCur)) {
            int32_t C = resultCls.at<int32_t>(targetY, targetX);
            if (C < 0)
              throw std::logic_error("Processed label of (targetY,targetX) should >= 0!"
                                     " in zxm::SegmentByLines()");
            connectedCls.emplace(C);
            if (C < minConnectedCls)
              minConnectedCls = C;
          }
        }
      }//for all adjacent pixel
      if (connectedCls.empty()) {
        //新类别
        mergeCls.tryInsertClass(nextCls);
        resultCls.at<int32_t>(i, j) = nextCls++;
      } else {
        if (connectedCls.size() > 1) {
          for (int C : connectedCls) {
            if (C != minConnectedCls)
              mergeCls.unionClass(minConnectedCls, C);
          }
        }
        resultCls.at<int32_t>(i, j) = minConnectedCls;
      }
    }//one pass for all pixels
  }
  //单独对边界上的像素点处理，赋予类别
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      int32_t cCur = clusterMap.at<int32_t>(i, j);
      int32_t curLine = lineMap.at<int32_t>(i, j);
      if (curLine <= 0)
        //只处理线段上的像素归属问题
        continue;
      //1、如果当前像素周围（4邻域）有cID一致的不在线上的项，则和其一致；
      //2、不满足1时，如果周围存在之前处理的cID一致且也在线上的项，则和其一致；
      //3、不满足2时，周围存在还未处理的cID一致且也在线上的项，暂时处理不了，创建新ID吧
      //4、否则（cID不一致），创建新ID。
      bool found = false;
      for (const Pxi32_t &yxOff: {Pxi32_t{0, -1},
                                  Pxi32_t{-1, 0},
                                  Pxi32_t{0, 1},
                                  Pxi32_t{1, 0}}) {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (0 <= targetY && targetY < Rows &&
            0 <= targetX && targetX < Cols) {
          int32_t cTarget = clusterMap.at<int32_t>(targetY, targetX);
          int32_t targetLine = lineMap.at<int32_t>(targetY, targetX);
          if (cTarget == cCur && targetLine <= 0) {
            resultCls.at<int32_t>(i, j) = resultCls.at<int32_t>(targetY, targetX);
            found = true;//break do-while
            break;//break for
          }
        }
      }
      if (found)
        continue;//go to next line pixel
      for (const Pxi32_t &yxOff: {Pxi32_t{0, -1},
                                  Pxi32_t{-1, 0}}) {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (0 <= targetY && targetY < Rows &&
            0 <= targetX && targetX < Cols) {
          int32_t cTarget = clusterMap.at<int32_t>(targetY, targetX);
          int32_t targetLine = lineMap.at<int32_t>(targetY, targetX);
          if (cTarget == cCur && targetLine > 0) {
            resultCls.at<int32_t>(i, j) = resultCls.at<int32_t>(targetY, targetX);
            found = true;//break do-while
            break;//break for
          }
        }
      }
      if (found)
        continue;//go to next line pixel
      //否则，创建新ID
      mergeCls.tryInsertClass(nextCls);
      resultCls.at<int32_t>(i, j) = nextCls++;
    }//one pass for line pixel
  }

  if (enableDebug)
    zxm::DrawClusters("../dbg/RawLineClusters.png", resultCls);

  std::vector<uint32_t> shrinkClass;
  int nCls = (int) mergeCls.shrink(&shrinkClass);
  // second pass
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      int32_t C = resultCls.at<int32_t>(i, j);
      if (C < 0)
        throw std::logic_error("Processed label should >= 0!"
                               " in zxm::SegmentByLines()");
      resultCls.at<int32_t>(i, j) = (int) shrinkClass[C];
    }
  }
  swap(clusterMap, resultCls);
  return nCls;
}
