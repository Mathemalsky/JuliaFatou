clang-format -i src/*.cpp include/*.hpp lib/*/src/*.cu lib/*/include/*.hpp

cd build
cmake ..
make
cd ..
