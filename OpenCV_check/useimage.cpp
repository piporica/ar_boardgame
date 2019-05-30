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

	//�չٴڰ���
	//�ϴܿ����� ������� ��;

	Point centerPoint; //�չٴ� �߽� ��ġ 
	double radius;


	Mat frame = imread("hand.jpg");

	cvtColor(frame, YCrCbframe, COLOR_BGR2YCrCb);
	split(YCrCbframe, planes); // �ɰ��� �ϰ� ������ �̷��� 

	mask = (minCr < planes[1]) & (planes[1] < maxCr) & (minCb < planes[2]) & (planes[2] < maxCb);


	Mat eroded;
	Mat closed;

	//�������� ���� Ŭ���� > �̷ε� 
	morphologyEx(mask, closed, MORPH_CLOSE, Mat(5, 5, CV_8U, Scalar(1)));
	erode(closed, eroded, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2); //ħ��

	//�Ÿ���ȯ �����ֱ�
	distanceTransform(eroded, dst, DIST_L2, DIST_MASK_PRECISE, 5);
	normalize(dst, dstshow, 0, 255, NORM_MINMAX, CV_8UC1);

	//�չٴ� �׸���
	centerPoint = getHandCenter1(eroded, radius);
	circle(frame, centerPoint, 2, Scalar(0, 255, 0), -1);
	circle(frame, centerPoint, (int)(radius + 0.5), Scalar(255, 0, 0), 2);

	//�հ��� ����1
	cout << getFingerNum1(eroded, centerPoint, radius, 1.8);
	
	DrawConvex(eroded);

	//�̹��� ����
	imshow("frame", frame);
	imshow("eroded", eroded);
	imshow("dstshow", dstshow);

	//imshow("clmg", clmg);

	waitKey(0);
		
}

//�չٴ� �߽� ���ϱ�1
Point getHandCenter1(const Mat & mask, double& radius) {

	//�Ÿ� ��ȯ ����� ������ ����
	Mat dst1;
	distanceTransform(mask, dst1, DIST_L2, 5);  //����� CV_32SC1 Ÿ��

	//�Ÿ� ��ȯ ��Ŀ��� ��(�Ÿ�)�� ���� ū �ȼ��� ��ǥ��, ���� ���´�.

	int maxIdx[2];    //��ǥ ���� ���� �迭(��, �� ������ �����)
	minMaxIdx(dst1, NULL, &radius, NULL, maxIdx, mask);   //�ּҰ��� ��� X

	return Point(maxIdx[1], maxIdx[0]);
}


//�հ��� ���� ���� - �� �׷��� �ϱ�
int getFingerNum1(const Mat & mask, Point center, double radius, double scale)
//scale : ������ ���� �������� ���� ���
{
	//�� Mat�� �� �׸���
	Mat clmg(mask.size(), CV_8U, Scalar(0));
	circle(clmg, center, radius * scale, Scalar(255));

	//vector�� ����
	vector <vector<Point>> contours;
	findContours(clmg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE); //�����

	if (contours.size() == 0) return -3; //�� ������ ã��x

	for (int i = 0; i < contours[0].size(); i++)
	{
		cout << i << ": ";
		cout << contours[0][i].x << ", ";
		cout << contours[0][i].y << endl;
	}

	//�ܰ����� ���� ���� mask�� ���� 0->1�� ���� Ȯ��
	int fingerCount = 0;

	for (int i = 1; i < contours[0].size(); i++) {
		Point p1 = contours[0][i - 1];
		Point p2 = contours[0][i];

		if (mask.at<uchar>(p1.y, p1.x) == 0 && mask.at<uchar>(p2.y, p2.x) > 1)
			fingerCount++;

	}

	return fingerCount - 1;
}



//�������� �̿��� �ո�� ����
void DrawConvex(const Mat& mask)
{
	//������ ����
	RNG rng(12345);

	vector <vector<Point>> contours;
	findContours(mask, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	//����ũ �����

	vector<vector<Point> >hull(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		convexHull(contours[i], hull[i]);
	}

	Mat drawing = Mat::zeros(mask.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++) // ���߿� ���� ū�Ÿ� �ؾ����� ������..........? �����̸�..........?
	{
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(drawing, contours, (int)i, color);
		drawContours(drawing, hull, (int)i, color);
	}

	imshow("Hull demo", drawing);

}