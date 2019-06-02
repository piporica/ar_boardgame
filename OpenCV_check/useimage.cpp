/**
  @file videocapture_basic.cpp
  @brief A very basic sample for using VideoCapture and VideoWriter
  @author PkLab.net
  @date Aug 24, 2016
*/
#include "header.h"


int main(int, char**)
{
	cvtColor(frame, YCrCbframe, COLOR_BGR2YCrCb);
	split(YCrCbframe, planes); // �ɰ��� �ϰ� ������ �̷��� 

	mask = (minCr < planes[1]) & (planes[1] < maxCr) & (minCb < planes[2]) & (planes[2] < maxCb);


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

	//�հ��� ����1 - 1������...
	//cout <<  getFingerNum1(eroded, centerPoint, radius, 1.8);
	
	//������ �׸���

	cvFillHoles(eroded);
	DrawConvex(eroded);

	//�̹��� ����

	imshow("mask", mask);
	imshow("frame", frame);
	imshow("eroded", eroded);

	//getRealcenterPoint();
	//imshow("clmg", clmg);

	waitKey(0);
		
}

//�չٴ� �߽� ���ϱ�1 - �ո�� ���� x
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
/*
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
*/


//�������� �̿��� �ո�� ���� 
void DrawConvex(const Mat& mask)
{
	findContours(mask, Contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>>hull(Contours.size());
	vector<vector<int> > hullsI(Contours.size()); // Indices to contour points
	vector<vector<Vec4i>> defects(Contours.size());

	//��������ȭ
	_hull = hull;
	_hullsI = hullsI;
	_defects = defects;

	//������ ����
	for (int i = 0; i < Contours.size(); i++)
	{
		convexHull(Contours[i], hull[i], false);
		convexHull(Contours[i], hullsI[i], false);
		if (hullsI[i].size() > 3) // You need more than 3 indices          
		{
			convexityDefects(Contours[i], hullsI[i], defects[i]);
		}
	}

	Mat drawing = Mat::zeros(mask.size(), CV_8UC3);

	/// Draw convexityDefects

	int count = 0;
	Point temparr[10]; //������ �հ�����������(�� �ȴٸ餾;)

	for (int i = 0; i < Contours.size(); ++i)
	{
		for (const Vec4i& v : defects[i])
		{
			Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
			drawContours(drawing, Contours, (int)i, color);
			drawContours(drawing, hull, (int)i, color);


			float depth = v[3];
			cout << depth << endl;
			if (depth > 2500) //  filter defects by depth, e.g more than 10
			{
				int startidx = v[0]; Point ptStart(Contours[i][startidx]);
				int endidx = v[1]; Point ptEnd(Contours[i][endidx]);
				int faridx = v[2]; Point ptFar(Contours[i][faridx]);

				line(drawing, ptStart, ptEnd, Scalar(255, 255, 0), 1);
				line(drawing, ptStart, ptFar, Scalar(255, 255, 0), 1);
				line(drawing, ptEnd, ptFar, Scalar(255, 255, 0), 1);

				circle(drawing, ptFar, 4, Scalar(0, 255, 255), 2);
				circle(drawing, ptStart, 4, Scalar(0, 0, 255), 2);
				circle(drawing, ptEnd, 4, Scalar(0, 0, 255), 2);
				temparr[count] = ptFar;
				count++;
			}
		}
	}
	vector<Point> defectPoints(count);
	for (int k = 0; k < count; k++)
	{
		defectPoints[k] = temparr[k];
	}
	_defectPoints = defectPoints;

	imshow("Hull demo", drawing);
}

void getRealcenterPoint()
{
	// �չٴ� �ٰ��� �׸���
	Mat check(mask.size(), CV_8U, Scalar(0));
	for (int i = 0; i < _defectPoints.size(); i++)
	{
		circle(check, _defectPoints[i], 4, Scalar(255), 2);
	}

	//imshow("check", check);
}

//�̹��� ���� �޲ٱ�
void cvFillHoles(Mat &input)
{
	cv::Mat image = input;

	cv::Mat image_thresh;
	cv::threshold(image, image_thresh, 125, 255, cv::THRESH_BINARY);

	// Loop through the border pixels and if they're black, floodFill from there
	cv::Mat mask;
	image_thresh.copyTo(mask);
	for (int i = 0; i < mask.cols; i++) {
		if (mask.at<char>(0, i) == 0) {
			cv::floodFill(mask, cv::Point(i, 0), 255, 0, 10, 10);
		}
		if (mask.at<char>(mask.rows - 1, i) == 0) {
			cv::floodFill(mask, cv::Point(i, mask.rows - 1), 255, 0, 10, 10);
		}
	}
	for (int i = 0; i < mask.rows; i++) {
		if (mask.at<char>(i, 0) == 0) {
			cv::floodFill(mask, cv::Point(0, i), 255, 0, 10, 10);
		}
		if (mask.at<char>(i, mask.cols - 1) == 0) {
			cv::floodFill(mask, cv::Point(mask.cols - 1, i), 255, 0, 10, 10);
		}
	}


	// Compare mask with original.
	cv::Mat newImage;
	image.copyTo(newImage);
	for (int row = 0; row < mask.rows; ++row) {
		for (int col = 0; col < mask.cols; ++col) {
			if (mask.at<char>(row, col) == 0) {
				newImage.at<char>(row, col) = 255;
			}
		}
	}
	cv::imshow("filled image", mask);
	input = newImage;
}