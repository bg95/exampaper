#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ImageUtil.h"

cv::Mat normalizeCharNLN(cv::Mat c, int maxx, int maxy);
Matrix normalizeCharNLN(Matrix img, int maxx, int maxy);
