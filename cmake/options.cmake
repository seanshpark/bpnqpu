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

#
# option for building
#

# platform in lower case
string(TOLOWER ${CMAKE_SYSTEM_NAME} PLATFORM_NAME_L)

# redirect if user sets output folder to specific place
if(DEFINED TARGET_OUTPUT)
  set(PROJ_LIB_OUT ${TARGET_OUTPUT})
  set(PROJ_BIN_OUT ${TARGET_OUTPUT})
else()
  set(_PROJ_BIN_ROOT ${CMAKE_BINARY_DIR})
  set(PROJ_LIB_OUT "${_PROJ_BIN_ROOT}/../.")
  set(PROJ_BIN_OUT "${_PROJ_BIN_ROOT}/../.")
endif()

# platform specific options
include("cmake/option/option_${TARGET_PLATFORM}.cmake")

# add user libsomething
set(PROJ_INCLUDES ${PROJ_INCLUDES} ${PROJ_USER_INCDIR})
set(PROJ_LIBPATHS ${PROJ_LIBPATHS} ${PROJ_USER_LIBDIR})
set(PROJ_LIBS     ${PROJ_LIBS}     ${PROJ_USER_LIBNAME})

# add common flags
foreach(FLAG ${FLAGS_COMMON})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FLAG}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
endforeach()

# ignore warings
set(ignoreUnusedWarnings "${CMAKE_TOOLCHAIN_FILE}")
