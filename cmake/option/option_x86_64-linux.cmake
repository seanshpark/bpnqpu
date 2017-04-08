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
# x86_64 linux compile options
#

message(FATAL_ERROR "Please build in ARM-Linux, i.e. Raspberry Pi")

# message("-- Building for x86-64 Linux")

# include linux common
# include("cmake/option/option_linux-common.cmake")

# no additional target includes
# set(TARGET_INC )

# user lib libsomething
# set(LIBUSER $ENV{HOME}/bin/libsomething)
# set(PROJ_USER_INCDIR ${LIBUSER}/include)
# set(PROJ_USER_LIBDIR ${LIBUSER}/lib)
# set(PROJ_USER_LIBNAME "something")
