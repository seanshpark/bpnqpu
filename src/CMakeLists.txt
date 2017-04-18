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
set(TARGETPRJ "bpnqpu")

# source file(s)
set(TARGETPRJ_SOURCEFILES
         helloworld.cpp)

# addition library
link_directories(${PROJ_LIBPATHS})

# build executable:
add_executable(${TARGETPRJ} ${TARGETPRJ_SOURCEFILES})
target_include_directories(${TARGETPRJ} SYSTEM PRIVATE ${TARGET_INC})
target_include_directories(${TARGETPRJ} PUBLIC ${PROJ_INCLUDES}
                           ${CMAKE_SOURCE_DIR}/libqpu/QPULib/Lib)
target_link_libraries(${TARGETPRJ} LINK_PUBLIC ${PROJ_LIBS} "qpu")
set_target_properties(${TARGETPRJ} PROPERTIES
                        ARCHIVE_OUTPUT_DIRECTORY "${PROJ_LIB_OUT}"
                        LIBRARY_OUTPUT_DIRECTORY "${PROJ_LIB_OUT}"
                        RUNTIME_OUTPUT_DIRECTORY "${PROJ_BIN_OUT}")

# copy data to execution folder
# configure_file(something.data ${PROJ_BIN_OUT}/something.data)
