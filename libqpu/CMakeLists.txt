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
set(QPUPROJECT "qpu")

set(QL "QPULib/Lib")

# source file(s)
set(QPUPROJECT_SOURCEFILES
        ${QL}/Kernel.cpp
        ${QL}/Source/Syntax.cpp
        ${QL}/Source/Int.cpp
        ${QL}/Source/Float.cpp
        ${QL}/Source/Stmt.cpp
        ${QL}/Source/Pretty.cpp
        ${QL}/Source/Translate.cpp
        ${QL}/Source/Interpreter.cpp
        ${QL}/Source/Gen.cpp
        ${QL}/Target/Syntax.cpp
        ${QL}/Target/SmallLiteral.cpp
        ${QL}/Target/Pretty.cpp
        ${QL}/Target/RemoveLabels.cpp
        ${QL}/Target/CFG.cpp
        ${QL}/Target/Liveness.cpp
        ${QL}/Target/RegAlloc.cpp
        ${QL}/Target/ReachingDefs.cpp
        ${QL}/Target/Subst.cpp
        ${QL}/Target/LiveRangeSplit.cpp
        ${QL}/Target/Satisfy.cpp
        ${QL}/Target/LoadStore.cpp
        ${QL}/Target/Emulator.cpp
        ${QL}/Target/Encode.cpp
        ${QL}/VideoCore/Mailbox.cpp
        ${QL}/VideoCore/Invoke.cpp
        ${QL}/VideoCore/VideoCore.cpp
        )

# addition library
link_directories(${PROJ_LIBPATHS})

# build executable: libqpu
add_library(${QPUPROJECT} STATIC ${QPUPROJECT_SOURCEFILES})
target_include_directories(${QPUPROJECT} SYSTEM PRIVATE ${TARGET_INC})
target_include_directories(${QPUPROJECT} PUBLIC ${PROJ_INCLUDES}
                           ${CMAKE_CURRENT_LIST_DIR}/${QL})
target_link_libraries(${QPUPROJECT} LINK_PUBLIC ${PROJ_LIBS})
set_target_properties(${QPUPROJECT} PROPERTIES
                        ARCHIVE_OUTPUT_DIRECTORY "${PROJ_LIB_OUT}"
                        LIBRARY_OUTPUT_DIRECTORY "${PROJ_LIB_OUT}"
                        RUNTIME_OUTPUT_DIRECTORY "${PROJ_BIN_OUT}")
