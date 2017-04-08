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
# armv7l linux compile options
#

message("-- Building for ARMv7l Linux")

# check native or cross build
if(NOT EXISTS "/lib/arm-linux-gnueabihf")
  # ROOTFS_ARM should be defined
  if(NOT EXISTS $ENV{ROOTFS_ARM})
    message(FATAL_ERROR "Please set ROOTFS_ARM environment to target ARM ROOTFS to build")
  endif()
endif()

# set linux common flags
include("cmake/option/option_linux-common.cmake")

# addition for arm-linux
set(FLAGS_COMMON ${FLAGS_COMMON}
                 "-mcpu=cortex-a7"
                 "-mfloat-abi=hard"
                 "-mfpu=neon-vfpv4")

# no additional target includes
set(TARGET_INC )

# user lib libsomething for arm
# set(LIBUSER $ENV{HOME}/bin/libsomething.arm)
# set(PROJ_USER_INCDIR ${LIBUSER}/include)
# set(PROJ_USER_LIBDIR ${LIBUSER}/lib)
# set(PROJ_USER_LIBNAME "something")
