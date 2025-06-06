#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <chrono>
#include <tuple>
#include <ctime>

#include "version.h"

int thermalPadding = 20; // do not read thermal data N pixels from the edge, higher value lowers processing time
constexpr int font = cv::FONT_HERSHEY_SIMPLEX;
const cv::Scalar white(255, 255, 255);
const cv::Scalar red(0, 0, 255);
const cv::Scalar green(0, 255, 0);
const cv::Scalar blue(255, 0, 0);
const cv::Scalar black(0, 0, 0);
constexpr float fontScale = 0.4;
constexpr int textBorderWidth = 3;
const std::vector<std::tuple<int, std::string>> colormaps = {
    {cv::ColormapTypes::COLORMAP_BONE, "Bone"},
    {cv::ColormapTypes::COLORMAP_TURBO, "Turbo"},
    {cv::ColormapTypes::COLORMAP_DEEPGREEN, "DeepGreen"},
    {cv::ColormapTypes::COLORMAP_OCEAN, "Ocean"},
    {cv::ColormapTypes::COLORMAP_HOT, "Hot"},
    {cv::ColormapTypes::COLORMAP_MAGMA, "Magma"},
    {cv::ColormapTypes::COLORMAP_INFERNO, "Inferno"},
    {cv::ColormapTypes::COLORMAP_TWILIGHT_SHIFTED, "TwilightShifted"},

    //{cv::ColormapTypes::COLORMAP_VIRIDIS, "Viridis"},
    //{cv::ColormapTypes::COLORMAP_CIVIDIS, "Cividis"},
    //{cv::ColormapTypes::COLORMAP_PINK, "Pink"},
    //{cv::ColormapTypes::COLORMAP_JET, "Jet"},
    //{cv::ColormapTypes::COLORMAP_HSV, "HSV"},
    //{cv::ColormapTypes::COLORMAP_AUTUMN, "Autumn"},
    //{cv::ColormapTypes::COLORMAP_WINTER, "Winter"},
    //{cv::ColormapTypes::COLORMAP_RAINBOW, "Rainbow"},
    //{cv::ColormapTypes::COLORMAP_SUMMER, "Summer"},
    //{cv::ColormapTypes::COLORMAP_SPRING, "Spring"},
    //{cv::ColormapTypes::COLORMAP_COOL, "Cool"},
    //{cv::ColormapTypes::COLORMAP_PARULA, "Parula"},
    //{cv::ColormapTypes::COLORMAP_PLASMA, "Plasma"},
    //{cv::ColormapTypes::COLORMAP_TWILIGHT, "Twilight"},
};

std::string elapsedTime(std::chrono::seconds elapsed) {
    const int hours = std::chrono::duration_cast<std::chrono::hours>(elapsed).count();
    elapsed -= std::chrono::hours(hours);
    const int minutes = std::chrono::duration_cast<std::chrono::minutes>(elapsed).count();
    elapsed -= std::chrono::minutes(minutes);
    const int seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    std::ostringstream oss;
    oss << "Rec:" << std::setfill('0') << std::setw(2) << hours << ":" << std::setfill('0') << std::setw(2) << minutes << ":" << std::setfill('0') << std::setw(2) << seconds;
    return oss.str();
}

std::string timestampFilename(const std::string& label) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    const std::tm tm = *std::localtime(&now_time);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S") << label;
    return oss.str();
}

void printFrameInfo(cv::Mat frame) {
    std::cout << "Rows: " << frame.rows << std::endl;
    std::cout << "Cols: " << frame.cols << std::endl;
    std::cout << "Channels: " << frame.channels() << std::endl;
    std::cout << "Depth: " << frame.depth() << std::endl;
    std::cout << "Type: " << frame.type() << std::endl;
    std::cout << "Element Size: " << frame.elemSize() << " bytes" << std::endl;
    std::cout << "Element Size (Channel): " << frame.elemSize1() << " bytes" << std::endl;
    std::cout << "Total Elements: " << frame.total() << std::endl;
    std::cout << "Is Empty: " << frame.empty() << std::endl;
    exit(0);
}

std::string getThermalValue(cv::Mat frame, int x, int y, bool conv) {
    const uint16_t* targetPixel = frame.ptr<uint16_t>(y, x);
    const uint16_t thermValueLow = targetPixel[0];
    const uint16_t thermValueHigh = targetPixel[1];
    std::string tempFormat;
    std::ostringstream oss;
    oss.precision(2);
    const float thermValue = (thermValueLow + thermValueHigh) / 2;
    const float cTemp = thermValue / 64 - 273.15;
    if (conv) {
        oss << std::fixed << (cTemp * 9 / 5) + 32 << " F";
    } else {
        oss << std::fixed << cTemp << " C";
    }
    // std::cout << thermValueLow << ", " << thermValueHigh << ", " << oss.str() << std::endl;
    return oss.str();
}

std::tuple<int, int, int, int> getValues(cv::Mat img) {
    int16_t highestValue = 0;
    int16_t lowestValue = 32767; // highest int16_t value
    int highestX;
    int highestY;
    int lowestX;
    int lowestY;
    for (int y = thermalPadding; y < img.rows - thermalPadding; ++y) {
        for (int x = thermalPadding; x < img.cols - thermalPadding; ++x) {
            const uint16_t pixelValue = img.at<uint16_t>(y, x);
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


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " -d <deviceInt>" << std::endl;
        return EXIT_FAILURE;
    }
    int deviceInt;
    if (argv[1][1] == 'd') {
        deviceInt = std::atoi(argv[2]);
    }
    std::cout << "cppThermalCamera v" << PROJECT_VERSION_MAJOR << "." << PROJECT_VERSION_MINOR << "." << PROJECT_VERSION_PATCH << std::endl;
    std::cout << R"(keymap:
     i  | toggle information (display [thermal area outline, colormap name])
     c  | toggle crosshair
     w  | toggle temp conversion
    h l | toggle High/Low points
     k  | toggle High/Low points temp labels
    z x | scale frame - +
    b n | thermal area - +
     m  | cycle through Colormaps
     p  | save frame to PNG file
    r t | record / stop
     q  | quit)" << std::endl;
    cv::VideoWriter videoWriter;
    char windowName[24];
    sprintf(windowName, "%s v%d.%d.%d", "cppThermalCamera", PROJECT_VERSION_MAJOR, PROJECT_VERSION_MINOR, PROJECT_VERSION_PATCH);
    auto recordingStartTime = std::chrono::system_clock::now();
    int mapInt = 0;
    int scale = 2;
    bool tempConv = true;
    bool recording = false;
    bool crosshair = true;
    bool info = false;
    bool lowPoint = false;
    bool highPoint = true;
    bool highLowLabel = false;
    int colormapsLen = static_cast<int>(colormaps.size());
    // set video source gstreamer to raw
    char pipeline[128];
    sprintf(pipeline, "v4l2src device=/dev/video%d ! video/x-raw, width=256, height=384, format=YUY2 ! appsink drop=1", deviceInt);
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the Thermal camera." << std::endl;
        return -1;
    }
    cv::Mat frame;
    // start video capture
    while (true) {
        // fps counter part0
        //int64 start = cv::getTickCount();
        // get frame
        cap >> frame;
        //printFrameInfo(frame); // testing
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
        //printFrameInfo(visibleMat);  // testing
        // Extract the first channel
        cv::Mat singleChannelMat;
        cv::extractChannel(visibleMat, singleChannelMat, 0);
        // convert 1 channel gray mat to RGB
        cv::Mat rgbFrame;
        cv::cvtColor(singleChannelMat, rgbFrame, cv::COLOR_GRAY2BGR);
        // get current colormap
        auto [colormapInt, colormapText] = colormaps[mapInt];
        // apply colormap
        cv::applyColorMap(singleChannelMat, singleChannelMat, colormapInt);
        // scale mat
        cv::Mat scaledImage;
        cv::resize(singleChannelMat, scaledImage, cv::Size(256 * scale, 192 * scale), 0, 0, cv::INTER_CUBIC);
        if (crosshair) {
            // draw crosshair
            cv::line(scaledImage, cv::Point((128 * scale) - 10, 96 * scale), cv::Point((128 * scale)+10, 96 * scale), red, 1);
            cv::line(scaledImage, cv::Point(128 * scale, (96 * scale) - 10), cv::Point(128 * scale, (96 * scale)+10), red, 1);
            // get center thermal value
            std::string centerThermalValue = getThermalValue(thermalMat, 128, 96, tempConv);
            // display thermal value
            cv::putText(scaledImage, centerThermalValue, cv::Point((256* scale)-65,(192 * scale)-4), font, fontScale, black, textBorderWidth);
            cv::putText(scaledImage, centerThermalValue, cv::Point((256* scale)-65,(192 * scale)-4), font, fontScale, white, 1);
        }
        if (highPoint || lowPoint) {
            auto [hX, hY, lX, lY] = getValues(thermalMat);
            if (highPoint) {
                // highest temp dot
                cv::circle(scaledImage, cv::Point(hX * scale, hY * scale), 1, black, 2);
                cv::circle(scaledImage, cv::Point(hX * scale, hY * scale), 1, red, 1);
                // get temp at highest point
                if (highLowLabel) {
                    std::string highestThermalValue = getThermalValue(thermalMat, hX, hY, tempConv);
                    // highest temp text
                    cv::putText(scaledImage, highestThermalValue, cv::Point((hX + 2) * scale, (hY + 7) * scale), font, fontScale, black, textBorderWidth);
                    cv::putText(scaledImage, highestThermalValue, cv::Point((hX + 2) * scale, (hY + 7) * scale), font, fontScale, white, 1);
                }
            }
            if (lowPoint) {
                // lowest temp dot
                cv::circle(scaledImage, cv::Point(lX * scale, lY * scale), 1, blue, 2);
                cv::circle(scaledImage, cv::Point(lX * scale, lY * scale), 1, white, 1);
                if (highLowLabel) {
                    // get lowest temp value
                    std::string lowestThermalValue = getThermalValue(thermalMat, lX, lY, tempConv);
                    // lowest value text
                    cv::putText(scaledImage, lowestThermalValue, cv::Point((lX + 2) * scale, (lY + 7) * scale), font, fontScale, black, textBorderWidth);
                    cv::putText(scaledImage, lowestThermalValue, cv::Point((lX + 2) * scale, (lY + 7) * scale), font, fontScale, white, 1);
                }
            }
        }
        // show thermal search area and colormap text
        if (info) {
            // display colormapText
            cv::putText(scaledImage, colormapText, cv::Point(0, 11), font, fontScale, black, textBorderWidth);
            cv::putText(scaledImage, colormapText, cv::Point(0, 11), font, fontScale, white, 1);
            // draw thermalSearchArea box
            cv::rectangle(scaledImage, cv::Point(thermalPadding*scale, thermalPadding*scale), cv::Point((256-thermalPadding)*scale, (192-thermalPadding)*scale), red, 1);
        }
        // Write frame to videoWriter and display elapsed time if recording
        if (recording) {
            // get elapsed time
            auto currentTime = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - recordingStartTime);
            std::string recElapsedTime = elapsedTime(elapsed);
            // Display elapsed time
            cv::putText(scaledImage, recElapsedTime, cv::Point((256 * scale) - 88, 11), font, fontScale, black, textBorderWidth);
            cv::putText(scaledImage, recElapsedTime, cv::Point((256 * scale) - 88, 11), font, fontScale, white, 1);
            videoWriter.write(scaledImage);
        }
        // show frame
        cv::imshow(windowName, scaledImage);
        // handle key presses
        switch (cv::waitKey(37)) {
            case 'q':
                return 0;
            case 'm':
                if (mapInt == colormapsLen-1) {
                    mapInt = 0;
                } else {
                    mapInt++;
                }
                break;
            case 'x':
                if (!recording) {
                    scale++;
                }
                break;
            case 'z':
                if (scale > 1 && !recording) {
                    scale--;
                }
                break;
            case 'p':
                if (!imwrite(timestampFilename(".png"), scaledImage)) {
                        std::cerr << "Could not save image!" << std::endl;
                        return -1;
                    }
                break;
            case 'w':
                tempConv = !tempConv;
                break;
            case 'r':
                recordingStartTime = std::chrono::system_clock::now();
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
                highPoint = !highPoint;
                break;
            case 'k':
                highLowLabel = !highLowLabel;
                break;
            case 'l':
                lowPoint = !lowPoint;
                break;
            case 'c':
                crosshair = !crosshair;
                break;
            case 'i':
                info = !info;
                break;
            case 'n':
                if (thermalPadding > 2) {
                    thermalPadding--;
                }
                break;
            case 'b':
                if (thermalPadding < 80) {
                    thermalPadding++;
                }
                break;
            default: break;
        }
        // fps counter part1
        //double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        //std::cout << "\rFPS : " << fps;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

