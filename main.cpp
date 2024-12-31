#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    int mapInt = 0;
    int scale = 2;
    std::vector<int> colormaps = {
        cv::ColormapTypes::COLORMAP_BONE,
        cv::ColormapTypes::COLORMAP_JET,
        cv::ColormapTypes::COLORMAP_RAINBOW,
        cv::ColormapTypes::COLORMAP_HSV,
        cv::ColormapTypes::COLORMAP_AUTUMN,
        cv::ColormapTypes::COLORMAP_WINTER,
        cv::ColormapTypes::COLORMAP_OCEAN,
        cv::ColormapTypes::COLORMAP_SUMMER,
        cv::ColormapTypes::COLORMAP_SPRING,
        cv::ColormapTypes::COLORMAP_COOL,
        cv::ColormapTypes::COLORMAP_PINK,
        cv::ColormapTypes::COLORMAP_HOT
    };
    int deviceInt{std::stoi(argv[1])};
    std::cout << "Opening device: " << deviceInt << std::endl;
    cv::VideoCapture cap(deviceInt); // 0 is the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the webcam." << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_CONVERT_RGB, false);
    cv::Mat frame(cv::Size(256, 384), CV_16UC2);
    //////// show thermal data window part 1
    // cv::namedWindow("thermalData", cv::WINDOW_NORMAL);
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Could not grab a frame." << std::endl;
            break;
        }
        cv::Rect topHalf(0, 0, frame.cols, frame.rows / 2);
        cv::Mat visibleMat = frame(topHalf);
        cv::Rect bottomHalf(0, frame.rows / 2 , frame.cols, frame.rows / 2);
        cv::Mat thermalMat = frame(bottomHalf);
        // Extract thermal data /////////////////////
        uint16_t centerPixel = thermalMat.at<uint16_t>(96, 128);
        uint16_t gray16leValue = centerPixel;
        float cTemp = gray16leValue / 64 - 273.15;
        float fTemp = (cTemp * 9 / 5) + 32;
        // end temp data extraction ////////////////////
        cv::Mat matRGB;
        cv::cvtColor(visibleMat, matRGB, cv::COLOR_YUV2BGR_YUYV);
        cv::Mat colormapped;
        cv::applyColorMap(matRGB, colormapped, colormaps[mapInt]);
        // scale mat
        cv::Mat scaledImage;
        cv::resize(colormapped, scaledImage, cv::Size(256 * scale, 192 * scale), 0, 0, cv::INTER_NEAREST);
        // Overlay elements
        std::string tempText = std::to_string(fTemp) + " F";
        // cv::putText(&mat, text, cv::Point(50, 180), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), size);
        cv::putText(scaledImage, tempText, cv::Point(5 * scale, 5 * scale), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 2);
        cv::putText(scaledImage, tempText, cv::Point(5 * scale, 5 * scale), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        cv::circle(scaledImage, cv::Point(128 * scale, 96 * scale), 1, cv::Scalar(0, 0, 0), 2);
        cv::circle(scaledImage, cv::Point(128 * scale, 96 * scale), 1, cv::Scalar(255, 255, 255), 1);
        cv::imshow("Webcam", scaledImage);
        //////// show thermal data window part 2
        // cv::Mat thermalDataMat;
        // cv::cvtColor(thermalMat, thermalDataMat, cv::COLOR_YUV2BGR_YUYV);
        // cv::circle(thermalDataMat, cv::Point(128, 96), 1, cv::Scalar(255, 255, 255), -1);
        // cv::imshow("thermalData", thermalDataMat);
        switch (cv::waitKey(30)) {
            case 'q':
                return 0;
                break;
            case 'm':
                std::cout << "Applying map: " << colormaps[mapInt] << std::endl;
                if (mapInt == 11) {
                    mapInt = 0;
                } else {
                    mapInt++;
                }
                break;
            case 'z':
                scale++;
                break;
            case 'x':
                if (scale > 1) {
                    scale--;
                }
                break;
            case 'p':
                if (!cv::imwrite("output.png", scaledImage)) {
                        std::cerr << "Could not save image!" << std::endl;
                        return -1;
                    }
                break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
