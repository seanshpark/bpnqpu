/*
 * Copyright 2017 saehie.park@gmail.com
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "QPULib.h"

// Define function that runs on the GPU.
void f_mulsum(Ptr<Float> w, Ptr<Float> i, Ptr<Float> o)
{
  o = o + (me() << 4);
  i = i + (me() << 4);
  w = w + (me() << 4);

  *o = *w * *i;
}

typedef Kernel<Ptr<Float>, Ptr<Float>, Ptr<Float> > mulsum_code;

class TestClass
{
public:
  TestClass(int n);
  void Construct();
  void Call_mulsum();

public:
  int N;

  SharedArray<float> * iarray;
  SharedArray<float> * oarray;
  SharedArray<float> * warray;

  mulsum_code func_mulsum_;
};

TestClass::TestClass(int n)
  : N(n)
  , func_mulsum_(compile(f_mulsum))
{
  func_mulsum_.setNumQPUs(16);
}

void TestClass::Construct()
{
  printf("N = %d\n", N);
  iarray = new SharedArray<float>(N);
  oarray = new SharedArray<float>(N);
  warray = new SharedArray<float>(N);
}

void TestClass::Call_mulsum()
{
  func_mulsum_(warray, iarray, oarray);
}

int main()
{
  // Construct kernel
  TestClass* testobj = new TestClass(64);
  testobj->Construct();

  float* fc1 = &(*(testobj->iarray))[0];
  float* fc2 = &(*(testobj->warray))[0];

  // Allocate and initialise array shared between ARM and GPU
  for (int i = 0; i < testobj->N; i++)
  {
    fc1[i] = 0.1f * (float)i;
    fc2[i] = 0.12f;
  }

  // Invoke the kernel and display the result
  testobj->Call_mulsum();

  for (int i = 0; i < testobj->N; i++)
  {
    printf("%03d: %f %f %f\n", i, (*(testobj->iarray))[i],
                                  (*(testobj->warray))[i],
                                  (*(testobj->oarray))[i]);
  }

  return 0;
}