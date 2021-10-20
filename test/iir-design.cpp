/*
Copyright 2019 Dario Mambro

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "oversimple/Hiir.hpp"
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace oversimple;

int main()
{
  for (int i = 0; i < 2; ++i) {
    auto preset = oversimple::iir::detail::getOversamplingPreset(i);
    cout << "preset " << i << ":\n";
    cout << preset.print();
    cout << "\n\n";
    auto groupDelay = preset.getGroupDelayGraph(20050).getGraph();
    ofstream file("groupDelay_" + std::to_string(i) + ".json");
    file << "{ \"groupDelay\": [ ";
    for (int i = 0; i < groupDelay.size(); ++i) {
      file << std::to_string(groupDelay[i]) << ", ";
    }
    file << std::to_string(groupDelay.back());
    file << "] } ";
    file.close();
  }
  return 0;
}
