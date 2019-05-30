/**
  @file videocapture_basic.cpp
  @brief A very basic sample for using VideoCapture and VideoWriter
  @author PkLab.net
  @date Aug 24, 2016
*/
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;
Point getHandCenter1(const Mat& mask, double& radius);
int getFingerNum1(const Mat& mask, Point center, double radius, double scale);
void DrawConvex(const Mat& mask);

int main(int, char**)
{

	Mat YCrCbframe;
	vector<Mat> planes;
	Mat mask;

	Mat dst;
	Mat dstshow;

	int minCr = 133;
	int maxCr = 180;
	int minCb = 77;
	int maxCb = 139;

	//손바닥감지
	//일단원으로 검출부터 함;

	Point centerPoint; //손바닥 중심 위치 
	double radius;


	Mat frame = imread("hand.jpg");

	cvtColor(frame, YCrCbframe, COLOR_BGR2YCrCb);
	split(YCrCbframe, planes); // 쪼개서 하고 싶으면 이렇게 

	mask = (minCr < planes[1]) & (planes[1] < maxCr) & (minCb < planes[2]) & (planes[2] < maxCb);


	Mat eroded;
	Mat closed;

	//모폴로지 연산 클로즈 > 이로드 
	morphologyEx(mask, closed, MORPH_CLOSE, Mat(5, 5, CV_8U, Scalar(1)));
	erode(closed, eroded, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2); //침식

	//거리변환 보여주기
	distanceTransform(eroded, dst, DIST_L2, DIST_MASK_PRECISE, 5);
	normalize(dst, dstshow, 0, 255, NORM_MINMAX, CV_8UC1);

	//손바닥 그리기
	centerPoint = getHandCenter1(eroded, radius);
	circle(frame, centerPoint, 2, Scalar(0, 255, 0), -1);
	circle(frame, centerPoint, (int)(radius + 0.5), Scalar(255, 0, 0), 2);

	//손가락 세기1
	cout << getFingerNum1(eroded, centerPoint, radius, 1.8);
	
	DrawConvex(eroded);

	//이미지 띄우기
	imshow("frame", frame);
	imshow("eroded", eroded);
	imshow("dstshow", dstshow);

	//imshow("clmg", clmg);

	waitKey(0);
		
}

//손바닥 중심 구하기1
Point getHandCenter1(const Mat & mask, double& radius) {

	//거리 변환 행렬을 저장할 변수
	Mat dst1;
	distanceTransform(mask, dst1, DIST_L2, 5);  //결과는 CV_32SC1 타입

	//거리 변환 행렬에서 값(거리)이 가장 큰 픽셀의 좌표와, 값을 얻어온다.

	int maxIdx[2];    //좌표 값을 얻어올 배열(행, 열 순으로 저장됨)
	minMaxIdx(dst1, NULL, &radius, NULL, maxIdx, mask);   //최소값은 사용 X

	return Point(maxIdx[1], maxIdx[0]);
}


//손가락 갯수 세기 - 원 그려서 하기
int getFingerNum1(const Mat & mask, Point center, double radius, double scale)
//scale : 검출할 원의 반지름에 쓰일 배수
{
	//새 Mat에 원 그리기
	Mat clmg(mask.size(), CV_8U, Scalar(0));
	circle(clmg, center, radius * scale, Scalar(255));

	//vector로 저장
	vector <vector<Point>> contours;
	findContours(clmg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE); //컨투어링

	if (contours.size() == 0) return -3; //손 없으면 찾지x

	for (int i = 0; i < contours[0].size(); i++)
	{
		cout << i << ": ";
		cout << contours[0][i].x << ", ";
		cout << contours[0][i].y << endl;
	}

	//외곽선을 따라 돌며 mask의 값이 0->1인 지점 확인
	int fingerCount = 0;

	for (int i = 1; i < contours[0].size(); i++) {
		Point p1 = contours[0][i - 1];
		Point p2 = contours[0][i];

		if (mask.at<uchar>(p1.y, p1.x) == 0 && mask.at<uchar>(p2.y, p2.x) > 1)
			fingerCount++;

	}

	return fingerCount - 1;
}



//컨벡스를 이용한 손모양 검출
void DrawConvex(const Mat& mask)
{
	//컨벡스 쓰기
	RNG rng(12345);

	vector <vector<Point>> contours;
	findContours(mask, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	//마스크 컨투어링

	vector<vector<Point> >hull(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		convexHull(contours[i], hull[i]);
	}

	Mat drawing = Mat::zeros(mask.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++) // 개중에 가장 큰거만 해야하지 않을까..........? 영상이면..........?
	{
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(drawing, contours, (int)i, color);
		drawContours(drawing, hull, (int)i, color);
	}

	imshow("Hull demo", drawing);

}