// Copyright (c) 2020 YuChen. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <gperftools/profiler.h>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <regex>
#include <string>
#include "cuda/common.h"
#include "cuda/inner.h"

/**
 * # 安装gperftools
 * apt-get install google-perftools
 * apt-get install libgoogle-perftools-dev
 * #include <gperftools/profiler.h>
 * target_link_libraries(target profiler)
 *
 * # 使用pprof输出文本分析报告
 * git clone https://github.com/gperftools/gperftools.git
 * export PATH=/work/gpu-learn/study/gperftools/src:$PATH
 * pprof --text ./demo case1.perf >case1.txt
 *
 * # 使用Graphviz和FlameGraph生成图形化分析报告
 * apt-get install graphviz
 * git clone https://github.com/brendangregg/FlameGraph.git
 * export PATH=/work/gpu-learn/study/FlameGraph:$PATH
 * pprof --svg ./demo case1.perf >case1.svg # 可视化图形报告
 * pprof --collapsed ./demo case1.perf > case1.cbt
 * flamegraph.pl case1.cbt > flame.svg # 绘制火焰图
 * flamegraph.pl --invert --color aqua case1.cbt > icicle.svg # 绘制冰柱图
*/

using namespace std::literals::string_literals;

int InnerMain(int argc, char** argv) {
  auto make_cpu_profiler =             // lambda表达式启动性能分析
      [](const std::string& filename)  // 传入性能分析的数据文件名
  {
    ProfilerStart(filename.c_str());  // 启动性能分析
    ProfilerRegisterThread();         // 对线程做性能分析

    return std::shared_ptr<void>(  // 返回智能指针
        nullptr,                   // 空指针，只用来占位
        [](void*) {                // 删除函数执行停止动作
          ProfilerStop();          // 停止性能分析
        });
  };

  auto make_regex = [](const auto& txt)  // 生产正则表达式
  { return std::regex(txt); };

  auto make_match = []()  // 生产正则匹配结果
  { return std::smatch(); };

  {
    auto cp = make_cpu_profiler("case1.perf");  // 启动性能分析
    auto str = "neir:automata"s;
    // 正则表达式对象，下面reg和what放入循环里面将会产生性能瓶颈
    auto reg = make_regex(R"(^(\w+)\:(\w+)$)");
    auto what = make_match();
    for (int i = 0; i < 1000; i++) {               // 循环一千次
      assert(regex_match(str, what, reg));  // 正则匹配
    }
  }
  return 0;
}
