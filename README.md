# SurveyPilot

**面向科研调研的本地工作台：检索论文、整理原文证据、核查引用并生成可继续编辑的综述草稿。**

SurveyPilot 将一次研究任务保存在独立会话中。你可以在浏览器里设置检索范围、查看每个执行阶段、检查证据矩阵与引用状态，并在任务中断后从保存的位置继续。系统也支持在已有会话中提出普通问题。

> 生成结果用于辅助研究。引用核查未通过时，报告会标为待核查草稿；用于论文或正式报告前仍需人工核对原文。

## 当前功能

| 环节 | 已实现的能力 |
| --- | --- |
| 论文发现 | 从 arXiv、OpenAlex、Semantic Scholar 检索；可限制年份、来源、结果数量、精读数量和排除词，并对候选论文去重。 |
| 引文扩展 | 可选填 DOI、arXiv 或 Semantic Scholar 种子论文，按设定层数和页数查找引用与参考文献，并展示本次取得的局部关系。 |
| 阅读与检索 | 先判断摘要相关性，再下载和解析可获取的 PDF；保留页码与较长上下文，使用 BM25、Chroma 向量检索和 RRF 融合查找片段。向量不可用时可退回关键词检索。 |
| 证据整理 | 提取 12 维实证矩阵，支持查看原句、编辑单元格、整表人工确认与独立模型复核；另生成跨论文比较线索。 |
| 综述写作 | 基于已阅读论文生成大纲与正文，进行引用核查，并在限定次数内修订。证据不足或引用未通过时保留草稿标识。 |
| 会话工作台 | 通过 SSE 展示阶段进度、工具调用、用量和产物；会话保存在本地 SQLite 与文件系统，支持停止和从检查点继续。 |
| 导出 | 可下载研究产物，以及 BibTeX、RIS 和中文 LaTeX 文件；LaTeX 导出保留报告的核查状态。 |

Docling 解析和 CrossEncoder 重排属于可选能力，需要另外安装依赖并准备相应模型。

## 工作流程

```mermaid
flowchart LR
    A[研究主题与检索条件] --> B[论文检索]
    B --> C[可选引文扩展]
    C --> D[摘要筛选与全文阅读]
    D --> E[实证矩阵与有限补搜]
    E --> F[可选人工确认和复核]
    F --> G[跨论文比较与综合分析]
    G --> H[大纲与正文]
    H --> I[引用核查与限次修订]
    I --> J[会话产物与导出]
```

如果检索没有可用论文，或全文未能建立可核查的片段，工作流会给出说明，不继续生成没有原文依据的综述。引文扩展、矩阵确认和补搜都可以按任务需要启用或跳过。

## 快速开始

需要 Python 3.12+、[uv](https://docs.astral.sh/uv/) 和 Node.js/npm。

```bash
git clone git@github.com:kultayf/SurveyPilot.git
cd SurveyPilot
uv sync
npm run front:install
cp config/model.example.json config/model.json
cp config/system.example.yaml config/system.yaml
```

在本地的 `config/model.json` 中配置模型 Provider、Agent 档位和 embedding 模型。也可以启动后在浏览器的「系统配置」页面填写并测试连接。论文检索密钥如需使用，填写在本地的 `config/system.yaml`；没有密钥时可尝试匿名检索，服务端可能限制请求频率。

分别启动后端和前端：

```bash
# 终端 1
uv run python main.py
```

```bash
# 终端 2
npm run front:dev
```

打开 <http://127.0.0.1:5173/>。后端默认运行在 `127.0.0.1:8000`，API 文档位于 <http://127.0.0.1:8000/docs>。

在「会话工作台」中新建会话，输入研究主题。展开检索设置可选择来源、年份、论文数量、引文扩展及写作前矩阵确认。运行时可查看进度；完成后从会话中打开证据和报告产物。

### 本地配置与数据

| 路径 | 用途 | Git 状态 |
| --- | --- | --- |
| `config/model.example.json` | 不含密钥的模型配置示例 | 提交到仓库 |
| `config/system.example.yaml` | 不含密钥的系统参数示例 | 提交到仓库 |
| `config/model.json` | 本地模型与 API 配置 | 已忽略 |
| `config/system.yaml` | 本地系统参数及可选论文来源密钥 | 已忽略 |
| `data/`、`logs/` | 会话、论文缓存、向量数据和日志 | 已忽略 |

**不要把模型 API 信息、Semantic Scholar 或 OpenAlex 密钥写入源码、示例文件、Issue 或提交记录。**

## 项目结构

```text
SurveyPilot/
├── config/              # 可公开的配置示例；本地配置由 Git 忽略
├── front/               # Vue 3 + TypeScript 工作台
├── src/
│   ├── agents/          # 检索、阅读、分析与写作 Agent
│   ├── api/             # FastAPI 接口
│   ├── graph/           # LangGraph 研究流程与检查点
│   ├── llm/             # 模型协议适配与配置
│   ├── paper_retrieval/ # 论文来源连接器
│   ├── repositories/    # 会话、设置与向量数据持久化
│   ├── services/        # 会话执行和系统设置
│   └── evaluation/      # 本地科研评测工具
├── test/                # 已有自动化测试与联调脚本
├── main.py              # 后端启动入口
└── pyproject.toml       # Python 依赖与可选功能
```

后端使用 FastAPI、LangGraph、SQLite、PyMuPDF4LLM 和 Chroma；前端使用 Vue 3、TypeScript 与 Vite。模型接口支持 OpenAI 兼容协议和 Anthropic Messages 协议。

## 使用边界

- Semantic Scholar 等外部数据源可能限流，论文元数据和全文可用性取决于来源服务。
- PDF 中的复杂表格、公式和缺字内容可能无法完整还原；引用核查也不能替代人工审稿。
- 跨论文比较输出的是供复核的线索，不代表整个研究领域的共识或空白。
- 本地数据和模型调用可能包含未公开的研究内容，请自行管理运行环境与供应商费用。

## 贡献

欢迎通过 [Issue 和 Pull Request](https://github.com/kultayf/SurveyPilot) 反馈问题或改进功能。
