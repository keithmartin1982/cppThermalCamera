#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <tuple>
#include <ctime>

const int thermalPadding = 5; // do not read thermal data N pixels from the edge
const int font = cv::FONT_HERSHEY_SIMPLEX;
const cv::Scalar white(255, 255, 255);
const cv::Scalar red(0, 0, 255);
const cv::Scalar green(0, 255, 0);
const cv::Scalar blue(255, 0, 0);
const cv::Scalar black(0, 0, 0);
const float fontScale = 0.4;
const int textBorderWidth = 3;
const std::vector<int> colormaps = {
    cv::ColormapTypes::COLORMAP_BONE,
    cv::ColormapTypes::COLORMAP_JET,
    cv::ColormapTypes::COLORMAP_DEEPGREEN,
    cv::ColormapTypes::COLORMAP_VIRIDIS,
    cv::ColormapTypes::COLORMAP_CIVIDIS,
    cv::ColormapTypes::COLORMAP_TURBO,
    cv::ColormapTypes::COLORMAP_TWILIGHT_SHIFTED,
    cv::ColormapTypes::COLORMAP_OCEAN,
    cv::ColormapTypes::COLORMAP_PINK,
    cv::ColormapTypes::COLORMAP_HOT,
    cv::ColormapTypes::COLORMAP_MAGMA,
    cv::ColormapTypes::COLORMAP_INFERNO,
    //cv::ColormapTypes::COLORMAP_HSV,
    //cv::ColormapTypes::COLORMAP_AUTUMN,
    //cv::ColormapTypes::COLORMAP_WINTER,
    //cv::ColormapTypes::COLORMAP_RAINBOW,
    //cv::ColormapTypes::COLORMAP_SUMMER,
    //cv::ColormapTypes::COLORMAP_SPRING,
    //cv::ColormapTypes::COLORMAP_COOL,
    //cv::ColormapTypes::COLORMAP_PARULA,
    //cv::ColormapTypes::COLORMAP_PLASMA,
    //cv::ColormapTypes::COLORMAP_TWILIGHT,
};

std::string timestampFilename(std::string label) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&now_time);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S") << label;
    return oss.str();
}

std::string getThermalValue(cv::Mat image, int x, int y, bool conv) {
    uint16_t* targetPixel = image.ptr<uint16_t>(y, x);
    uint16_t thermValueLow = targetPixel[0];
    uint16_t thermValueHigh = targetPixel[1];
    std::string tempFormat;
    std::ostringstream oss;
    oss.precision(2);
    float vTemp;
    float thermValue = (thermValueLow + thermValueHigh) / 2;
    float cTemp = thermValue / 64 - 273.15;
    if (conv) {
        vTemp = (cTemp * 9 / 5) + 32;
        oss << std::fixed << vTemp << " F";
    } else {
        oss << std::fixed << cTemp << " C";
    }
    return oss.str();
}

std::tuple<int, int, int, int>  getValues(cv::Mat img) {
    int16_t highestValue = 0;
    int16_t lowestValue = 32767; // highest int16_t value
    int highestX;
    int highestY;
    int lowestX;
    int lowestY;
    for (int y = thermalPadding; y < img.rows - thermalPadding; ++y) {
        for (int x = thermalPadding; x < img.cols - thermalPadding; ++x) {
            uint16_t pixelValue = img.at<uint16_t>(y, x);
            if (pixelValue < lowestValue) {
                lowestValue = pixelValue;
                lowestX = x;
                lowestY = y;
            }
            if (pixelValue > highestValue) {
                highestValue = pixelValue;
                highestX = x;
                highestY = y;
            }
        }
    }
    return std::make_tuple(highestX, highestY, lowestX, lowestY);
}

int main(int argc, char **argv) {
    std::cout << R"(keymap:
     c  | toggle crosshair
     w  | toggle temp conversion
     h  | toggle hud
    z x | scale image + -
     m  | cycle through Colormaps
     p  | save frame to file
    r t | record / stop
     q  | quit
     )" << std::endl;
    cv::VideoWriter videoWriter;
    int mapInt = 0;
    int scale = 2;
    bool tempConv = true;
    bool recording = false;
    bool crosshair = true;
    bool hud = true;
    int colormapsLen = static_cast<int>(colormaps.size());
    int deviceInt{std::stoi(argv[1])};
    //std::cout << "Opening device: " << deviceInt << std::endl;
    cv::VideoCapture cap(deviceInt);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the webcam." << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_CONVERT_RGB, false);
    cv::Mat frame(cv::Size(256, 384), CV_16UC2);
    // start video capture
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Could not grab a frame." << std::endl;
            break;
        }
        // get thermal mat
        cv::Rect bottomHalf(0, frame.rows / 2 , frame.cols, frame.rows / 2);
        cv::Mat thermalMat = frame(bottomHalf);
        // get visible mat
        cv::Rect topHalf(0, 0, 256, 192);
        cv::Mat visibleMat = frame(topHalf);
        // convert to rgb
        cv::Mat matRGB;
        cv::cvtColor(visibleMat, matRGB, cv::COLOR_YUV2BGR_YUYV);
        // apply colormap
        cv::Mat colormapped;
        cv::applyColorMap(matRGB, colormapped, colormaps[mapInt]);
        // scale mat
        cv::Mat scaledImage;
        cv::resize(colormapped, scaledImage, cv::Size(256 * scale, 192 * scale), 0, 0, cv::INTER_CUBIC);
        if (crosshair) {
            // line H
            //cv::line(scaledImage, cv::Point((128 * scale)-10, 96 * scale), cv::Point((128 * scale)+10, 96 * scale), cv::Scalar(0, 0, 0), 2);
            cv::line(scaledImage, cv::Point((128 * scale)-10, 96 * scale), cv::Point((128 * scale)+10, 96 * scale), red, 1);
            // line V
            //cv::line(scaledImage, cv::Point(128 * scale, (96 * scale) - 10), cv::Point(128 * scale, (96 * scale)+10), white, 2);
            cv::line(scaledImage, cv::Point(128 * scale, (96 * scale) - 10), cv::Point(128 * scale, (96 * scale)+10), red, 1);
            // get center thermal value
            std::string centerThermalValue = getThermalValue(thermalMat, 128, 96, tempConv);
            // display thermal value
            cv::putText(scaledImage, centerThermalValue, cv::Point((256* scale)-65,(192 * scale)-4), font, fontScale, black, textBorderWidth);
            cv::putText(scaledImage, centerThermalValue, cv::Point((256* scale)-65,(192 * scale)-4), font, fontScale, white, 1);
        }
        if (hud) {
            // Get hi low locations
            auto [hX, hY, lX, lY] = getValues(thermalMat);
            // lowest temp dot
            cv::circle(scaledImage, cv::Point(lX * scale, lY * scale), 1, blue, 2);
            cv::circle(scaledImage, cv::Point(lX * scale, lY * scale), 1, white, 1);
            // get lowest temp value
            std::string lowestThermalValue = getThermalValue(thermalMat, lX, lY, tempConv);
            // lowest value text
            cv::putText(scaledImage, lowestThermalValue, cv::Point((lX + 2) * scale, (lY + 10) * scale), font, fontScale, black, textBorderWidth);
            cv::putText(scaledImage, lowestThermalValue, cv::Point((lX + 2) * scale, (lY + 10) * scale), font, fontScale, white, 1);
            // highest temp dot
            cv::circle(scaledImage, cv::Point(hX * scale, hY * scale), 1, black, 2);
            cv::circle(scaledImage, cv::Point(hX * scale, hY * scale), 1, red, 1);
            // get temp at highest point
            std::string highestThermalValue = getThermalValue(thermalMat, hX, hY, tempConv);
            // highest temp text
            cv::putText(scaledImage, highestThermalValue, cv::Point((hX + 2) * scale, (hY + 10) * scale), font, fontScale, black, textBorderWidth);
            cv::putText(scaledImage, highestThermalValue, cv::Point((hX + 2) * scale, (hY + 10) * scale), font, fontScale, white, 1);
        }
        // Write frame to videoWriter if recording
        if (recording) {
            videoWriter.write(scaledImage);
        }
        // show frame
        cv::imshow("Webcam", scaledImage);
        switch (cv::waitKey(10)) {
            case 'q':
                return 0;
                break;
            case 'm':
                if (mapInt == colormapsLen-1) {
                    mapInt = 0;
                } else {
                    mapInt++;
                }
                break;
            case 'z':
                if (!recording) {
                    scale++;
                }
                break;
            case 'x':
                if (scale > 1 && !recording) {
                    scale--;
                }
                break;
            case 'p':
                if (!cv::imwrite(timestampFilename(".png"), scaledImage)) {
                        std::cerr << "Could not save image!" << std::endl;
                        return -1;
                    }
                break;
            case 'w':
                if (tempConv) {
                    tempConv = false;
                } else {
                    tempConv = true;
                }
                break;
            case 'r':
                videoWriter.open(timestampFilename(".avi"), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(256*scale, 192*scale));
                if (!videoWriter.isOpened()) {
                    std::cerr << "Could not open the output video file for write\n";
                    break;
                }
                std::cout << "Recording started..." << std::endl;
                recording = true;
                break;
            case 't':
                videoWriter.release();
                std::cout << "Recording stopped..." << std::endl;
                recording = false;
                break;
            case 'h':
                if (hud) {
                    hud = false;
                } else {
                    hud = true;
                }
                break;
            case 'c':
                if (crosshair) {
                    crosshair = false;
                } else {
                    crosshair = true;
                }
                break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
