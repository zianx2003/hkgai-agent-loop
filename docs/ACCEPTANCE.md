# 分阶段验收与演示

所有命令均在项目根目录运行，并使用已安装 `.[dev]` 的虚拟环境。
Windows 未激活环境时，把 `python` 替换为 `.\.venv\Scripts\python.exe`。

## Phase 1：完整 Agent Loop

```powershell
python -m hkgai_agent
```

离线 fake model 运行三轮：选择并加载 Skill → 批量读取两个文件 → 最终回答。
工具结果显示原始调用 ID，结束状态为 `final`。模型响应是脚本，工具通过真实
executor 执行。真实模型验收见 Phase 5。

## Phase 2：压缩前后结构

```powershell
python -m hkgai_agent.demos compression
```

上下文上限设为 2048 个估算 token，阈值为 35%，保留最近四条消息且不拆开工具交互。
本地运行结果：**1295 → 237 个估算 token，9 → 6 条消息**。

```text
压缩前：
  system                 原始系统指令
  user                   早期任务、预算和只读约束
  assistant [old-read]   较早工具请求
  tool      [old-read]   对应结果
  assistant              较早任务进度
  user                   最新用户请求
  assistant [recent-list]
  tool      [recent-list]
  assistant              下一步计划

压缩后：
  system                 原始系统指令，逐字保留
  system [summary]       任务、预算 <=10、禁止修改、已完成读取、待验证状态
  user                   最新用户请求，逐字保留
  assistant [recent-list]
  tool      [recent-list]
  assistant              下一步计划
```

JSON 报告逐条展示角色、调用 ID、摘要标记和内容；过长历史内容明确标为截断预览。
`checks` 验证保留条件与工具配对，任何不满足都不能得到 `passed: true`。
这里的摘要由 fake model 固定返回，用于验证压缩机制与状态保留检查；真实模型的
摘要质量不在这个离线演示中得到保证。

## Phase 3：两个延迟工具确实并发

```powershell
python -m hkgai_agent.demos parallel
```

`slow_lookup` 等待 0.3 秒后返回 `notes`；`fast_lookup` 等待 0.2 秒后返回
`constraints`。同样的两个调用分别在并发上限 1 和 2 下执行。

本地一次实际测量：

| 模式 | 总耗时 | 工具执行重叠 |
| --- | ---: | ---: |
| 顺序，concurrency=1 | 0.513 秒 | 0 秒 |
| 并行，concurrency=2 | 0.314 秒 | 0.202 秒 |

报告包含每个工具实际的开始和结束时间，以及 `slow-id → notes`、
`fast-id → constraints` 的结果对应关系。自动测试重新计算执行区间的交集，
同时检查较快工具先完成时结果仍对应原始调用 ID。时间会因机器负载变化，
不把固定加速比例当作正确性条件。

## Phase 4：code_review 与安全边界

```powershell
python -m hkgai_agent.demos code-review
```

Agent 加载 `skills/code_review/SKILL.md`，然后读取 `review_sample.py`。
样例故意包含空列表除零与可变默认参数两个问题，只被作为文本读取，不会导入或执行。
输出中的代码审查结论明确标记为脚本示例。

真实工具执行器同时验证：

- `read_file("../outside.txt")` 返回 `unsafe_path`。
- 未获 allowlist 授权的 `write_file` 返回 `not_allowed`。
- 未提供的 `shell` 返回 `not_allowed`。
- 样例源文件保持原样；Skill 的文字不能改变执行权限。

超时、部分失败、并发上限和共享资源读写隔离由自动测试覆盖。

## Phase 5：自动测试、真实 API、GitHub

```powershell
python -m pip check
python -m pytest -q -ra
python -m hkgai_agent.demos all --output artifacts/acceptance.json
```

`all` 只运行上述三个离线验收演示；生成的 JSON 明确包含
`live_api_executed: false`。依赖检查与离线演示已通过。

2026-10-19 本地 Windows / Python 3.11 验证结果：**141 项通过，2 项跳过**。
跳过项是工作区与 Skill 的符号链接逃逸测试：当前 Windows 权限不允许创建
测试用符号链接。普通路径越界、绝对路径、驱动器路径和遍历拒绝测试均已运行。

在同一个终端配置实际服务的 `OPENAI_MODEL` 和所需 `OPENAI_API_KEY`，必要时设置
`OPENAI_BASE_URL`，再执行：

```powershell
python -m hkgai_agent.demos live --output artifacts/live.json
```

真实 API 验收要求：模型通过工具实际加载 `code_review`、实际读取样例文件，
然后返回最终回答；所有工具调用成功且文件未变。缺少工具调用、API 错误、超时、
迭代耗尽均不能通过，不会自动退回 fake model。该命令会产生 API 请求及可能的费用。
验收检查证明调用流程；回答内容的质量仍需要人工阅读。

**当前真实 API 状态：未执行。** 本机未配置模型与 API 凭据；已验证缺少配置时
明确报错。测试中的 HTTPX MockTransport 只证明适配器与整个调用链可重复运行，
不能替代真实服务验收。

**GitHub 目标：[zianx2003/hkgai-agent-loop](https://github.com/zianx2003/hkgai-agent-loop)。**
本地 CI 文件已准备。推送后应确认远程分支与本地提交一致，并在 Actions 页面检查
GitHub 托管 CI 的实际运行结果。后续命令见 [GitHub 交付说明](GITHUB.md)。
