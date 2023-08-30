#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;


Mat SpaceGaussianKernel(Size_<uchar> kernel_size, double sigma_space)
{
    Mat kernel(kernel_size, CV_32FC1);

    int k_height = kernel_size.height;
    int k_width = kernel_size.width;

    int x = k_width / 2;
    int y = k_height / 2;

    for (int row = 0; row < k_height; row++)
    {
        for (int col = 0; col < k_width; col++)
        {
            double value_x = pow(col - x, 2) / (2 * pow(sigma_space, 2));
            double value_y = pow(row - y, 2) / (2 * pow(sigma_space, 2));

            double value = cv::exp(-(value_x + value_y));

            kernel.at<float>(row, col) = value;
        }
    }

    return kernel;

}

vector<float> ColorDisTable(double sigma_color)
{
   vector<float> colorDisTable(256,0);
   int n=0;
   for(int i=0;i<256;i++){
      colorDisTable[i] = exp(-pow(i,2)/(2*pow(sigma_color,2)));
   }
   return colorDisTable;
}

void MyBilateralFilter(Mat& input_img, Mat& output_img, Size_<uchar> kernel_size, double sigma_color, double sigma_space)
{
    sigma_color = sigma_color * sigma_color;
    sigma_space = sigma_space * sigma_space;

    input_img.copyTo(output_img);

    int height = input_img.rows;
    int width = input_img.cols;
    int channel = input_img.channels();

    int k_height = kernel_size.height;
    int k_width = kernel_size.width;
    vector<float> colDisTab;
    colDisTab = ColorDisTable(sigma_color);
    cv::copyMakeBorder(input_img, input_img, k_height / 2, k_height / 2, k_width / 2, k_width / 2, BORDER_CONSTANT);

    Mat space_kernel, bilateral_kernel;
    space_kernel = SpaceGaussianKernel(kernel_size, sigma_space);

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {

            for (int c = 0; c < channel; c++)
            {
                float anchor_pixel = input_img.at<Vec<uchar, 3>>(row + k_height / 2, col + k_width / 2)[c];

                space_kernel.copyTo(bilateral_kernel);
                for (int k_y = 0; k_y < k_height; k_y++)
                {
                    for (int k_x = 0; k_x < k_width; k_x++)
                    {
                        float pixel = input_img.at<Vec<uchar, 3>>(row + k_y, col + k_x)[c];
                        bilateral_kernel.at<float>(k_y, k_x) *= colDisTab[abs(pixel-anchor_pixel)];
                        //bilateral_kernel.at<float>(k_y, k_x) *= exp(-pow(pixel - anchor_pixel, 2) / (2 * pow(sigma_color, 2)));
                        //bilateral_kernel.at<float>(k_y, k_x) *= 1;

                    }
                }

                bilateral_kernel /= cv::sum(bilateral_kernel)[0];

                float result = 0;
                for (int k_y = 0; k_y < k_height; k_y++)
                {
                    for (int k_x = 0; k_x < k_width; k_x++)
                    {
                        float pixel = input_img.at<Vec<uchar, 3>>(row + k_y, col + k_x)[c];

                        result += bilateral_kernel.at<float>(k_y, k_x) * pixel;

                    }
                }

                output_img.at<Vec<uchar, 3>>(row, col)[c] = result;

            }

        }
    }

}

int main(void)
{
    Mat img = imread("test.bmp");
    Mat output_img, noisy_img;

    Mat noise = Mat::zeros(img.size(),img.type());
    randn(noise,(0,0,0),(10,10,10));
    add(noise,img,noisy_img);

    MyBilateralFilter(noisy_img, output_img, Size_<int>(3, 3), 5, 5);

    cv::imwrite("output_img.bmp", output_img);

    system("PAUSE");
    return 0;
}

//http://people.csail.mit.edu/sparis/bf/
