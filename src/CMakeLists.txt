# Copyright 2017 saehie.park@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# project name
set(HELLOWORLD "helloworld")

# source file(s)
set(HELLOWORLD_SOURCEFILES
        helloworld.cpp)

# addition library
link_directories(${PROJ_LIBPATHS})

# build executable: helloworld
add_executable(${HELLOWORLD} ${HELLOWORLD_SOURCEFILES})
target_include_directories(${HELLOWORLD} SYSTEM PRIVATE ${TARGET_INC})
target_include_directories(${HELLOWORLD} PUBLIC ${PROJ_INCLUDES}
                           ${CMAKE_SOURCE_DIR}/libqpu/QPULib/Lib)
target_link_libraries(${HELLOWORLD} LINK_PUBLIC ${PROJ_LIBS} "qpu")
set_target_properties(${HELLOWORLD} PROPERTIES
                        ARCHIVE_OUTPUT_DIRECTORY "${PROJ_LIB_OUT}"
                        LIBRARY_OUTPUT_DIRECTORY "${PROJ_LIB_OUT}"
                        RUNTIME_OUTPUT_DIRECTORY "${PROJ_BIN_OUT}")

# copy data to execution folder
# configure_file(something.data ${PROJ_BIN_OUT}/something.data)
