#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <time.h>
#include <ply.hpp>

#include <tr1/functional>
#include <tr1/tuple>

#include "opencv2/video/tracking.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define PI 3.14159265

using namespace cv;
using namespace std;

vector<Point> ve;
vector<Point> fc;
vector< vector<Point> > af;
Mat output;
Mat color;
float mx,my,mr,mg,mb;

void vertex_x_callback(ply::float32 x)
{
   mx = x*11 + 400;
   std::cout << x;
}

void vertex_y_callback(ply::float32 y)
{
   my = y*11 + 300;
   std::cout << " " << y;
}

void vertex_z_callback(ply::float32 z)
{
   ve.push_back(Point(mx,my));
   std::cout << " " << z << "\n";
}

void vertex_r_callback(ply::uint8 r)
{  
   mr = r > 5 ? r : 5;
   mr = 255 - r;
   std::cout << int(r);
}

void vertex_g_callback(ply::uint8 g)
{
   mg = g > 5 ? g : 5;
   mg = 255 - g;
   std::cout << " " << int(g);
}

void vertex_b_callback(ply::uint8 b)
{
   mb = b > 5 ? b : 5;
   mb = 255 - mb;
   circle(color,Point(mx,my),5,Scalar(mb,mg,mr),-1,8,0);
   std::cout << " " << int(b) << "\n";
}

void face_vertex_indices_begin(ply::uint8 size)
{
   fc.clear();   
   std::cout << "\n";
}

void face_vertex_indices_element(ply::uint32 vertex_index)
{
   fc.push_back(ve[vertex_index]);
   std::cout << vertex_index << " ";
}

void face_vertex_indices_end()
{
   af.clear();
   af.push_back(vector<Point>(fc));
   std::cout << "||";
   fillPoly(output,af,Scalar(255,255,255));
}

template  <typename ScalarType>
std::tr1::function <void (ScalarType)> scalar_property_definition_callback(const std::string& element_name, const std::string& property_name);

template <typename SizeType, typename ScalarType> std::tr1::tuple<std::tr1::function<void (SizeType)>, std::tr1::function<void (ScalarType)>, std::tr1::function<void ()> > list_property_definition_callback(const std::string& element_name, const std::string& property_name);

template  <>
std::tr1::function <void (ply::float32)> scalar_property_definition_callback(const std::string& element_name, const std::string& property_name)
{
   if (element_name == "vertex") {
      if (property_name == "x") {
         return vertex_x_callback;
      }
      else if (property_name == "y") {
         return vertex_y_callback;
      }
      else if (property_name == "z") {
         return vertex_z_callback;
      }
      else {
         return 0;
      }
   }
   else {
      return 0;
   }
}

template  <>
std::tr1::function <void (ply::uint8)> scalar_property_definition_callback(const std::string& element_name, const std::string& property_name)
{
   if (element_name == "vertex") {
      if (property_name == "red") {
         return vertex_r_callback;
      }
      else if (property_name == "green") {
         return vertex_g_callback;
      }
      else if (property_name == "blue") {
         return vertex_b_callback;
      }
      else {
         return 0;
      }
   }
   else {
      return 0;
   }
}

template <>
std::tr1::tuple<std::tr1::function<void (ply::uint8)>, std::tr1::function<void (ply::uint32)>, std::tr1::function<void ()> > list_property_definition_callback(const std::string& element_name, const std::string& property_name)
{
      std::cout << "test\n";
   if ((element_name == "face") && (property_name == "vertex_indices")) {
      return std::tr1::tuple<std::tr1::function<void (ply::uint8)>, std::tr1::function<void (ply::uint32)>, std::tr1::function<void ()> >(face_vertex_indices_begin,face_vertex_indices_element,face_vertex_indices_end);
   }
   else {
      return std::tr1::tuple<std::tr1::function<void (ply::uint8)>, std::tr1::function<void (ply::uint32)>, std::tr1::function<void ()> >(0, 0, 0);
   }
}

int main(int argc, char* argv[])
{
   output = Mat::zeros(Size(800,600),CV_8UC3);
   color = Mat::zeros(Size(800,600),CV_8UC3);
   
   int sqs = 20;
   circle(output,Point((800+sqs)/2,0),1,Scalar(255,255,255),-1,8,0);
   circle(output,Point((800-sqs)/2,0),1,Scalar(255,255,255),-1,8,0);
   
     
   ply::ply_parser ply_parser;
   
   ply::ply_parser::scalar_property_definition_callbacks_type scalar_property_definition_callbacks;
   ply::at <ply::float32>(scalar_property_definition_callbacks) = scalar_property_definition_callback <ply::float32>;
   ply::at <ply::uint8>(scalar_property_definition_callbacks) = scalar_property_definition_callback <ply::uint8>;
   ply_parser.scalar_property_definition_callbacks(scalar_property_definition_callbacks);
   
   
   ply::ply_parser::list_property_definition_callbacks_type list_property_definition_callbacks;
   ply::at<ply::uint8, ply::uint32>(list_property_definition_callbacks) = list_property_definition_callback<ply::uint8, ply::uint32>;
   ply_parser.list_property_definition_callbacks(list_property_definition_callbacks);

   ply_parser.parse(argv[1]);

   color.copyTo(output,color.mul(output/255));
      //Mat ROI = output(Rect(350,0,100,600));
   //ROI.copyTo(output(Rect(150,0,100,600)));
   //ROI.copyTo(output(Rect(550,0,100,600)));
   //ROI = output(Rect(0,275,800,50));
   //ROI.copyTo(output(Rect(0,225,800,50)));
   //ROI.copyTo(output(Rect(0,175,800,50)));
   //ROI.copyTo(output(Rect(0,125,800,50)));
   //ROI.copyTo(output(Rect(0,325,800,50)));
   //ROI.copyTo(output(Rect(0,375,800,50)));
   //ROI.copyTo(output(Rect(0,425,800,50)));
   circle(color,Point((800+sqs)/2,0),3,Scalar(0,255,0),-1,8,0);
   circle(color,Point((800-sqs)/2,0),3,Scalar(0,255,0),-1,8,0);
   output=Scalar(255,255,255)-output;
   imwrite(argv[2],output);
}
