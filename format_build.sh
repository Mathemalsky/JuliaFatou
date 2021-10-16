clang-format -i src/*.cu src/*.cpp include/*.hpp

cd build
cmake ..
make
cd ..
