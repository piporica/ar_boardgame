#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <cmath>
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

Point centerPoint; //�չٴ� �߽� ��ġ 
double radius;

Mat frame = imread("./hand-imgs/hand-1-1.jpg");
Mat eroded;
Mat closed;

RNG rng(12345);

vector <vector<Point>> Contours;


/*
	vector<vector<Point>>hull(Contours.size());
	vector<vector<int> > hullsI(Contours.size()); // Indices to contour points
	vector<vector<Vec4i>> defects(Contours.size());
	�̰� �����ͷ� �ϴ°� ���� ������?
*/

vector<vector<Point>>_hull;
vector<vector<int> > _hullsI;
vector<vector<Vec4i>> _defects;
//���� �� ������ �ε��� 
int LongestContour;

//����� ������/defect
vector<Point> _selectContours;
vector<Vec4i> _selectdefects;

vector<Point> _Fars;
vector<Point> _SEPoints;

////

Point getHandCenter1(const Mat& mask, double& radius);
//int getFingerNum1(const Mat& mask, Point center, double radius, double scale);
void DrawConvex(const Mat& mask);

void getRealcenterPoint(Mat& input);


void cvFillHoles(Mat& input);