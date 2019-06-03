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
	split(YCrCbframe, planes); // 쪼개서 하고 싶으면 이렇게 

	mask = (minCr < planes[1]) & (planes[1] < maxCr) & (minCb < planes[2]) & (planes[2] < maxCb);


	//모폴로지 연산 클로즈 > 이로드 
	morphologyEx(mask, closed, MORPH_CLOSE, Mat(5, 5, CV_8U, Scalar(1)));
	erode(closed, eroded, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2); //침식

	//구멍 채우기
	cvFillHoles(eroded);
	
	//컨벡스 그리기
	DrawConvex(eroded);

	//마스크 구하기
	Mat check(mask.size(), CV_8U, Scalar(0));
	getRealcenterPoint(); 
	
	//거리변환 보여주기
	distanceTransform(eroded, dst, DIST_L2, DIST_MASK_PRECISE, 5);
	normalize(dst, dstshow, 0, 255, NORM_MINMAX, CV_8UC1);

	//손바닥 그리기
	centerPoint = getHandCenter1(eroded, radius);
	circle(frame, centerPoint, 2, Scalar(0, 255, 0), -1);
	circle(frame, centerPoint, (int)(radius + 0.5), Scalar(255, 0, 0), 2);

	//손가락 세기1 - 1차실패...
	//cout <<  getFingerNum1(eroded, centerPoint, radius, 1.8);
	

	DrawRealConvex(eroded);
	//이미지 띄우기

	//imshow("check", check); 
	//imshow("mask", mask);
	//imshow("frame", frame);
	//imshow("eroded", eroded);

	//imshow("clmg", clmg);

	waitKey(0);
		
}

//손바닥 중심 구하기1 - 손목과 구분 x
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
/*
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
*/


//컨벡스를 이용한 손모양 검출 
void DrawConvex(const Mat& mask)
{
	findContours(mask, Contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>>hull(Contours.size());
	vector<vector<int> > hullsI(Contours.size()); // Indices to contour points
	vector<vector<Vec4i>> defects(Contours.size());



	LongestContour = 0;
	//컨벡스 쓰기
	for (int i = 0; i < Contours.size(); i++)
	{
		//제일 긴 컨투어 라인 찾기 
		if (Contours[LongestContour].size() < Contours[i].size())
			LongestContour = i;

		convexHull(Contours[i], hull[i], false);
		convexHull(Contours[i], hullsI[i], false);
		if (hullsI[i].size() > 3) // You need more than 3 indices          
		{
			convexityDefects(Contours[i], hullsI[i], defects[i]);
		}
	}
	cout << "골라진 컨투어 인덱스 : " << LongestContour << endl;


	Mat drawing = Mat::zeros(mask.size(), CV_8UC3);

	//전역변수화
	_hull = hull;
	_hullsI = hullsI;
	_defects = defects;
	
	_selectContours = Contours[LongestContour];
	_selectdefects = defects[LongestContour];
	_selecthull = hull[LongestContour];

	/// Draw convexityDefects

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

			}
		}
	}
	//imshow("Hull demo", drawing);
}

void getRealcenterPoint()
{
	int sumFarCenterX = 0, sumFarCenterY = 0;
	int sumSECenterX = 0, sumSECenterY = 0;

	int count = 0;
	Point selectFars[10];


	for (int i = 0; i < _selectdefects.size(); i++)
	{

		const Vec4i& v = _selectdefects[i];
		float depth = v[3];
		cout << depth << endl;
		if (depth > 1500) //  filter defects by depth, e.g more than 10
		{
			int startidx = v[0]; Point ptStart(_selectContours[startidx]);
			int endidx = v[1]; Point ptEnd(_selectContours[endidx]);
			int faridx = v[2]; Point ptFar(_selectContours[faridx]);

			selectFars[count] = ptFar;
			sumFarCenterX += ptFar.x;
			sumFarCenterY += ptFar.y;

			sumSECenterX += (ptStart.x + ptEnd.x);
			sumSECenterY += (ptStart.y + ptEnd.y);

			count ++;
		}
	}
	FarCenter.x = sumFarCenterX / count;
	FarCenter.y = sumFarCenterY / count;

	SECenter.x = sumSECenterX / (count * 2);
	SECenter.y = sumSECenterY / (count * 2);

	vector<Point> Fars(count);

	maxdist = 0;
	int Fdistance = 0;
	int FmaxdistIndex = 0;

	for (int i = 0; i < count; i++)
	{
		Fars[i] = selectFars[i];
	
		//하는김에 평균과 가장 먼 거리의 점 구하기
		Fdistance = sqrt(pow(Fars[i].x - FarCenter.x, 2) + pow(Fars[i].y - FarCenter.y, 2));
		if (Fdistance > maxdist)
		{
			FmaxdistIndex = i;
			maxdist = Fdistance;
		}
	}
	_Fars = Fars;

}

void DrawRealConvex(Mat& input)
{
	//기본 그리기
	Mat drawing = Mat::zeros(input.size(), CV_8UC3);

	//drawContours(drawing, Contours, (int)LongestContour, Scalar(100,100,100));
	drawContours(drawing, _hull, (int)LongestContour, Scalar(255, 255, 255));

	Point hull_center;
	int x = 0, y = 0;
	
	float scale = 1.2; // 교차점이 0개면 늘려야 함

	circle(drawing, FarCenter, 4, Scalar(255, 0, 0));
	circle(drawing, _Fars[FmaxdistIndex], 8, Scalar(255,0,0));
	
	

	cout << _selecthull.size();

	for (int i = 0; i < _selecthull.size(); i++)
	{
		circle(drawing, _selecthull[i], 4, Scalar(255, 0, 0));
		x += _selecthull[i].x;
		y += _selecthull[i].y;
	}

	hull_center.x = x / _selecthull.size();
	hull_center.y = y / _selecthull.size();

	circle(drawing, hull_center, 4, Scalar(255, 255, 255));


	//교차점 구하기 
	vector<Point> creossPoints(50);
	int crosscount = checkcross(scale, creossPoints);
	while (crosscount < 2)
	{
		cout << scale;
		scale += 0.3;
		crosscount = checkcross(scale, creossPoints);
	}

	circle(drawing, FarCenter, maxdist * scale, Scalar(255, 100, 0));
	imshow("test", drawing);
}

int checkcross(int scale, vector<Point> &crossPoints)
{
	Point prevPoint;
	Point nowPoint;

	int countcross = 0;
	double prevdist = 0;
	double nowdist = 0;

	for (int j = 0; j < _selecthull.size(); j++)
	{
		//직전은 원 안/ 직후는 원 밖(또는 반대)인 점들 찾기
		//맨끝점이랑 첫점도 체크
		if (j == 0)
		{
			prevPoint = _selecthull[_selecthull.size() - 1];
			nowPoint = _selecthull[j];
		}
		else
		{
			prevPoint = _selecthull[j - 1];
			nowPoint = _selecthull[j];
		}

		prevdist = sqrt(pow(prevPoint.x - FarCenter.x, 2) + pow(prevPoint.y - FarCenter.y, 2));
		nowdist = sqrt(pow(nowPoint.x - FarCenter.x, 2) + pow(nowPoint.y - FarCenter.y, 2));

		if (((maxdist * scale) - prevdist) * ((maxdist * scale) - nowdist) < 0)
		{
			crossPoints[countcross*2] = prevPoint;
			crossPoints[countcross*2+1] = nowPoint;
			countcross++;
		}
		crossPoints.resize(countcross * 2);
	}

	cout << "교차점 : " << countcross;
	return countcross;

}
//이미지 구멍 메꾸기
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
	//cv::imshow("filled image", mask);
	input = newImage;
}