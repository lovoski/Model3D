# Introduction
This is 3D geometry processing library written by Prof Xin. I organized it with cmake.

## Usage
Place the whole folder in the root folder of you code.
Include this to your repo with the following code in your main CMakeLists.txt.
```cmake
add_subdirectory(Model3D)
include_directories(
  ${PROJECT_SOURCE_DIR}/Model3D/include
  # ...
)
# ...
target_link_library(${PROJECT_NAME} PRIVATE Model3D)
```


## License
This code is licensed under MIT license.