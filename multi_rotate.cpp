#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI

using namespace std;
using namespace cv;
namespace fs = std::filesystem;  
void draw_progress(float progress)
{
    std::cout << "[";
    int bar_width = 70;
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

// XYZ-eular rotation 
Mat eular2rot(Vec3d theta)
{
    // Calculate rotation about x axis
    Mat R_x = (Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );
     
    // Calculate rotation about y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );
     
    // Calculate rotation about z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);
     
    // Combined rotation matrix
    Mat R = R_x * R_y * R_z;
     
    return R;
}

// Rotation matrix to rotation vector in XYZ-eular order
Vec3d rot2eular(Mat R)
{
    double sy = sqrt(R.at<double>(2,2) * R.at<double>(2,2) +  R.at<double>(1,2) * R.at<double>(1,2) );
 
    bool singular = sy < 1e-6; // If
 
    double x, y, z;
    if (!singular)
    {
        x = atan2(-R.at<double>(1,2) , R.at<double>(2,2));
        y = atan2(R.at<double>(0,2), sy);
        z = atan2(-R.at<double>(0,1), R.at<double>(0,0));
    }
    else
    {
        x = 0;
        y = atan2(R.at<double>(0,2), sy);
        z = atan2(-R.at<double>(0,1), R.at<double>(0,0));
    }
    return Vec3d(x, y, z);
}

// rotate pixel, in_vec as input(row, col)
Vec2i rotate_pixel(const Vec2i& in_vec, Mat& rot_mat, int width, int height)
{
    Vec2d vec_rad = Vec2d(M_PI*in_vec[0]/height, 2*M_PI*in_vec[1]/width);

    Vec3d vec_cartesian;
    vec_cartesian[0] = -sin(vec_rad[0])*cos(vec_rad[1]);
    vec_cartesian[1] = sin(vec_rad[0])*sin(vec_rad[1]);
    vec_cartesian[2] = cos(vec_rad[0]);

    double* rot_mat_data = (double*)rot_mat.data;
    Vec3d vec_cartesian_rot;
    vec_cartesian_rot[0] = rot_mat_data[0]*vec_cartesian[0] + rot_mat_data[1]*vec_cartesian[1] + rot_mat_data[2]*vec_cartesian[2];
    vec_cartesian_rot[1] = rot_mat_data[3]*vec_cartesian[0] + rot_mat_data[4]*vec_cartesian[1] + rot_mat_data[5]*vec_cartesian[2];
    vec_cartesian_rot[2] = rot_mat_data[6]*vec_cartesian[0] + rot_mat_data[7]*vec_cartesian[1] + rot_mat_data[8]*vec_cartesian[2];

    Vec2d vec_rot;
    vec_rot[0] = acos(vec_cartesian_rot[2]);
    vec_rot[1] = atan2(vec_cartesian_rot[1], -vec_cartesian_rot[0]);
    if(vec_rot[1] < 0)
        vec_rot[1] += M_PI*2;

    Vec2i vec_pixel;
    vec_pixel[0] = height*vec_rot[0]/M_PI;
    vec_pixel[1] = width*vec_rot[1]/(2*M_PI);

    return vec_pixel;
}

Mat rotateImage(const Mat& image, double roll, double pitch, double yaw)
{
    double im_width = image.cols;
    double im_height = image.rows;
    double im_size = im_width * im_height;
    Size im_shape(im_height, im_width);

    Mat2i im_pixel_rotate(im_height, im_width);
    Mat im_out(image.rows, image.cols, image.type());
    Vec3b* im_data = (Vec3b*)image.data;
    Vec3b* im_out_data = (Vec3b*)im_out.data;
    Mat rot_mat = eular2rot(Vec3f(RAD(roll), RAD(pitch), RAD(yaw)));

    for (int i = 0; i < static_cast<int>(im_height); i++)
    {
        for (int j = 0; j < static_cast<int>(im_width); j++)
        {
            Vec2i vec_pixel = rotate_pixel(Vec2i(i, j), rot_mat, im_width, im_height);
            int origin_i = vec_pixel[0];
            int origin_j = vec_pixel[1];
            if ((origin_i >= 0) && (origin_j >= 0) && (origin_i < im_height) && (origin_j < im_width))
            {
                im_out_data[i * image.cols + j] = im_data[origin_i * image.cols + origin_j];
            }
        }
    }

    return im_out;
}

void saveRotatedImages(const string& filename, const string& outputFolder)
{
    Mat im = imread(filename);
    if (im.data == NULL)
    {
        cout << "Can't open image" << endl;
        return;
    }

    double startAngle = 0.0;
    double endAngle = 360.0;
    double angleStep = 30.0;

    string basename = filesystem::path(filename).stem().string();
    string folderPath = outputFolder + "/" + basename;
    filesystem::create_directory(folderPath);

    for (double angle = startAngle; angle <= endAngle; angle += angleStep)
    {
        Mat rotatedImage = rotateImage(im, 0.0, angle, 0.0);

        string angleStr = to_string(angle);
        angleStr.erase(angleStr.find_last_not_of('0') + 1, string::npos); // Remove trailing zeros
        angleStr.erase(angleStr.find_last_not_of('.') + 1, string::npos); // Remove decimal point if no fractional part

        string savePath = folderPath + "/" + basename + "_rotate_" + angleStr + ".jpg";
        imwrite(savePath, rotatedImage);
    }

    cout << "Rotated images saved in: " << folderPath << endl;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cout << "Usage: Equirectangular_rotate.out <Image file name> <Output folder>" << endl;
        return 0;
    }

    string filename = argv[1];
    string outputFolder = argv[2];

    saveRotatedImages(filename, outputFolder);

    return 0;
}
