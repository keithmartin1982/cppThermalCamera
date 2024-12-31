export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=:/usr/local/lib/pkgconfig/
 # shellcheck disable=SC2046
LDFLAGS=-static g++ main.cpp -o cppThermalCamera `pkg-config --cflags --libs opencv4`  -I/usr/local/include/opencv4/ && ./cppThermalCamera 2