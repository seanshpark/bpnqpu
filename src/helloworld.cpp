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
void hello(Ptr<Float> w, Ptr<Float> i, Ptr<Float> o)
{
  o = o + (me() << 4);
  i = i + (me() << 4);
  w = w + (me() << 4);

  *o = *w * *i;
}

int main()
{
  // Construct kernel
  auto k = compile(hello);
  int N;

  N = 32;

  // Allocate and initialise array shared between ARM and GPU
  SharedArray<float> iarray(N);
  SharedArray<float> oarray(N);
  SharedArray<float> warray(N);
  for (int i = 0; i < N; i++)
  {
    iarray[i] = 0.2f;
    warray[i] = 0.12f;
  }

  k.setNumQPUs(16);

  // Invoke the kernel and display the result
  k(&warray, &iarray, &oarray);
  for (int i = 0; i < N; i++)
  {
    printf("%03d: %f %f %f\n", i, iarray[i], warray[i], oarray[i]);
  }

  return 0;
}