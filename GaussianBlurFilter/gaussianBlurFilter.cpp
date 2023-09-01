#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;


vector<float> SpaceGaussianKernel(int kernel_size, double sigma_space)
{
   vector<float> kernel;
   double value,sum = 0;
   int x = kernel_size/2;

    for (int i = 0; i < kernel_size; i++)
    {
        value = pow(i - x, 2) / (2 * pow(sigma_space, 2));
        value = cv::exp(-(value));
        kernel.push_back(value);
        sum += value;
    }

    for(int i=0;i<kernel_size;i++)
    {
       kernel[i] /= sum;
    }

    return kernel;
}


void MyGaussianBlurFilter(Mat& input_img, Mat& output_img, Size_<uchar> kernel_size, double sigma_space)
{
    sigma_space = sigma_space * sigma_space;

    output_img = Mat::zeros(input_img.size(),input_img.type());

    int height = input_img.rows;
    int width = input_img.cols;
    int channel = input_img.channels();
    int k_height = kernel_size.height;
    int k_width = kernel_size.width;
    int i,j,k,row,col;
    vector<vector<float>> pBGR;
    vector<float> space_kernel_x,space_kernel_y;
    vector<float> temp(channel);

    cv::copyMakeBorder(input_img, input_img, k_height / 2, k_height / 2, k_width / 2, k_width / 2, BORDER_CONSTANT);
    space_kernel_y = SpaceGaussianKernel(k_height, sigma_space);
    space_kernel_x = SpaceGaussianKernel(k_width, sigma_space);

    for (row = 0; row < height; row++)
    {
        pBGR.resize(k_width,vector<float>(channel,0));
        for(i=0;i<k_height;i++){
           for(j=0;j<k_width-1;j++){
              for(k=0;k<channel;k++)
                 pBGR[j+1][k] += space_kernel_y[i]*input_img.at<Vec<uchar,3>>(row+i,j)[k];
           }
        }

        for (col = 0; col < width; col++)
        {
            memset(&temp[0],0,temp.size()*sizeof(temp[0]));
            for(i=0;i<k_width-1;i++){
              for(k=0;k<channel;k++){
                pBGR[i][k] = pBGR[i+1][k];
              }
            }
            for(k=0;k<channel;k++){
              for(i=0;i<k_height;i++){
                temp[k] += space_kernel_y[i]*input_img.at<Vec<uchar,3>>(row+i,col+k_width-1)[k];
              }
              pBGR[k_width-1][k]=temp[k];
            }
            memset(&temp[0],0,temp.size()*sizeof(temp[0]));
            for(k=0;k<channel;k++){
              for(j=0;j<k_width;j++){
                temp[k] += space_kernel_x[j]*pBGR[j][k];
              }
                output_img.at<Vec<uchar, 3>>(row, col)[k] = temp[k];
            }
        }
    }
}

int main(void)
{
    Mat img = imread("test.bmp");
    Mat output_img,noisy_img;

    Mat noise = Mat::zeros(img.size(),img.type());
    randn(noise,(0,0,0),(10,10,10));
    add(noise,img,noisy_img);
    MyGaussianBlurFilter(noisy_img, output_img, Size_<int>(3, 3), 5);

    cv::imwrite("output_img.bmp", output_img);

    system("PAUSE");
    return 0;
}

