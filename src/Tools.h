//
// Created by zxm on 2022/12/5.
//

#ifndef ELSED_TOOLS_H
#define ELSED_TOOLS_H

#include <cassert>
#include <cstdint>
#include <vector>
#include <map>
#include <execution>

#include <opencv2/core.hpp>


namespace zxm
{


//******************* 基本数学工具 ******************//


/**
 * 检测数学运算是否有错误，如果有则抛出异常
 */
void CheckMathError();


/**
* @brief 并查集
*/
struct ClassUnion {
  //parent[i] 表示类别i的父类别
  std::vector<uint32_t> parent;

  bool containClass(uint32_t c) {
    return parent.size() > c && parent[c] != uint32_t(-1);
  }

  uint32_t findRootClass(uint32_t c) {
    if (containClass(c))
      while (c != parent[c])
        c = parent[c];
    else
      c = (uint32_t) -1;
    return c;
  }

  bool tryInsertClass(uint32_t c) {
    if (c >= parent.size())
      parent.resize(c + 1, uint32_t(-1));
    if (!containClass(c)) {
      parent[c] = c;
      return true;
    } else
      // this ID is existed.
      return false;
  }

  void unionClass(uint32_t c1, uint32_t c2) {
    tryInsertClass(c1);
    tryInsertClass(c2);
    c1 = findRootClass(c1);
    c2 = findRootClass(c2);
    if (c1 <= c2)
      parent[c2] = c1;
    else
      parent[c1] = c2;
  }

  uint32_t shrink(std::vector<uint32_t> *outShrinkClass = nullptr) {
    //合并类别中间的空白。
    // 在所有类型合并结束后，每个类别的根类别可能出现[2,2,2,0,0,5,5,5]这种情况，
    // 将其收缩为 [0,0,0,1,1,2,2,2]
    std::vector<uint32_t> shrinkClass(parent.size(), (uint32_t) -1);
    uint32_t startShrinkCls = 0;
    std::map<uint32_t, uint32_t> rootClsToShrinkCls;
    for (uint32_t i = 0, iEnd = (uint32_t) parent.size(); i < iEnd; ++i) {
      if (!containClass(i))
        //并查集中没有类别i
        continue;
      uint32_t rootC = findRootClass(i);
      parent[i] = rootC;
      if (rootClsToShrinkCls.count(rootC))
        shrinkClass[i] = rootClsToShrinkCls.at(rootC);
      else {
        rootClsToShrinkCls[rootC] = startShrinkCls;
        shrinkClass[i] = startShrinkCls;
        ++startShrinkCls;
      }
    }
    if (outShrinkClass)
      outShrinkClass->swap(shrinkClass);
    // -1表示没有根类别
    return startShrinkCls;
  }
};


template<typename T>
inline
T cvMax(const cv::Mat &m) {
  T maxV = std::numeric_limits<T>::min();
  std::for_each(std::execution::seq,
                m.begin<T>(),
                m.end<T>(),
                [&maxV](const T &v) {
                  if (v > maxV)
                    maxV = v;
                });
  return maxV;
}


template<typename T>
inline
T clamp(T v, T minV, T maxV) {
  return v < minV ? minV : (v > maxV ? maxV : v);
}


template<typename T>
void shuffle(std::vector<T> &vs) {
  for (int i = (int) vs.size() - 1; i >= 0; --i) {
    int j = rand() % (i + 1);
    if (i != j)
      std::swap(vs[i], vs[j]);
  }
}


//******************* 图片读取工具 ******************//


/**
 * 将CV_32F的法向图（含零向量）修改为CV_32FC3格式（不含零向量）
 * @note 函数内部会对法向做归一化
 * @param m
 * @param mask
 * @return
 */
cv::Mat CV_Convert32FTo32FC3(const cv::Mat &m, const cv::Mat &mask);


cv::Mat CV_Resize256x192(const cv::Mat &m, int interpolation);


cv::Mat CV_Imread1920x1440(const std::string &file, int imreadFlag, int interpolation);


//******************* 图片保存工具 ******************//


void SampleAColor(double *color, double x, double min, double max);


void CV_ImWriteWithPath(const std::string &path, const cv::Mat &im);


cv::Mat DrawClusters(const std::string &savePath, const cv::Mat &clustersMap);


cv::Mat DrawNormals(const std::string &savePath, const cv::Mat &normals);


}

#endif //ELSED_TOOLS_H
