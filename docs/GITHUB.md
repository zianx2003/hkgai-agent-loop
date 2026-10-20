# GitHub 交付说明

目标仓库：[zianx2003/hkgai-agent-loop](https://github.com/zianx2003/hkgai-agent-loop)。

当前已准备好源码、示例工具、两个 Skill、自动测试、分阶段验收命令，以及
`.github/workflows/tests.yml`。工作流在 Ubuntu/Windows 和 Python 3.11/3.13
组合下运行测试与离线演示，不需要 API 密钥。推送后在仓库 Actions 页面确认
GitHub 托管的运行结果；本地测试通过不等同于远程 CI 通过。

## 本地检查

在仓库根目录的已激活虚拟环境中执行：

```powershell
python -m pip install -e ".[dev]"
python -m pip check
python -m pytest -q -ra
python -m hkgai_agent
python -m hkgai_agent.demos all --output artifacts/acceptance.json
```

真实 API 验收需要在同一个终端环境中设置 `OPENAI_MODEL` 和服务需要的
`OPENAI_API_KEY`，兼容服务还可设置 `OPENAI_BASE_URL`：

```powershell
python -m hkgai_agent.demos live --output artifacts/live.json
```

该命令会实际调用模型，必须成功加载 `code_review`、读取样例文件并给出最终回答
才算通过。尚未配置凭据时不要把模拟 HTTP 测试当成真实 API 验收。

## 初始化与推送

先用 `git status` 和 `git remote -v` 检查已有状态。仅在目录尚未初始化、且远程
仓库为空时，才执行下列首次初始化步骤。已有 Git 仓库不要重复初始化；远程已有
提交时应先获取并审查其历史，不要强制覆盖。

```powershell
git init -b main
git add .gitignore .gitattributes pyproject.toml README.md src tests examples skills docs .github
git status --short
git diff --cached --stat
git commit -m "Implement modular async agent loop and acceptance demos"
git remote add origin https://github.com/zianx2003/hkgai-agent-loop.git
git push -u origin main
```

日常修改只需检查状态、暂存所需文件、创建提交，再执行 `git push origin main`。

`.gitignore` 排除虚拟环境、构建产物、缓存、环境文件、常见私钥与凭据文件、
本机代理配置目录以及 `artifacts/`。
生成的验收报告可以留作本地演示材料。提交前检查暂存文件；不要提交真实凭据。
项目目前没有添加许可证，发布时由项目所有者决定是否需要及使用哪种许可证。
