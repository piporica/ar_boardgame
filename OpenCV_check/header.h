#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

Mat YCrCbframe;
vector<Mat> planes;
Mat mask;

Mat dst;
Mat dstshow;

int minCr = 133; //128 
int maxCr = 180;
int minCb = 78; //73
int maxCb = 139;

Point centerPoint; //손바닥 중심 위치 
double radius;

Mat frame = imread("./hand-imgs/nonthumb.jpg");
Mat eroded;
Mat closed;

RNG rng(12345);

vector <vector<Point>> Contours;

/*
	vector<vector<Point>>hull(Contours.size());
	vector<vector<int> > hullsI(Contours.size()); // Indices to contour points
	vector<vector<Vec4i>> defects(Contours.size());
	이거 포인터로 하는게 낫지 않을까?
*/

vector<vector<Point>>_hull;
vector<vector<int> > _hullsI;
vector<vector<Vec4i>> _defects;
vector<Point> _defectPoints;

////

Point getHandCenter1(const Mat& mask, double& radius);
//int getFingerNum1(const Mat& mask, Point center, double radius, double scale);
void DrawConvex(const Mat& mask);

void getRealcenterPoint();


void cvFillHoles(Mat& input);