# 真实系统检验表（计划 §3.4，任务 3.4）

证据截至 2026-09-28（网页、公开仓库）；T1、T3、T4 的测量格按阶段 2 测试集（20 世界，`runs/stage2_e2/`、`runs/stage2_e3/`）于 2026-09-29 更新，Mem0 的 T3、T4 按 E5（15 个 3 个月测试世界，`runs/stage3_e5/`）更新；本仓库代码按 2026-09-29 的 HEAD `56876e3` 复核。机器可读版本：`system_audit.json`（每条记录含理由、判定依据与证据指针）；证据清单：`evidence.jsonl`（每行一条，含 URL 或路径、访问日期、支持的判断、被哪些系统引用）。

## 判定口径

- **D、B** 按计划 §2.2 严格定义。D=1 要求类型化值 + 源片段，可由确定性检查器复核；"部分" = 有类型化结构或源引用，但无片段级确定性复核；0 = 值只在自然语言里。B=1 要求键同时含 (源纪元或读集, κ 版本, θ_w)，任一分量变化即失效；键不完整一律记 0，并在 `B_key_components` 里写出实际有的分量（如"只有源"）；"仅时间有效性" = 只在新的矛盾信息到来时失效，不随源修订、定义或写入模型变化失效。
- **T1–T4** 取值：通过 / 不通过 / 不可判定 / 待 E2、E3、E5、E6。**通过**只用于有测量或按构造成立（无派生状态）的格。**不通过**只用于文档明确确立的格（如根本不能导出记忆）或推断理由写明的格。**不可判定**用于外部无法运行、文档也不能确立结论的格：厂商或论文用其他指标、其他世界报告的数字只写进理由，不据以判通过。
- **T2 的具体规则**："不通过" = 文档表明没有完整的记忆导出（只能让模型用散文复述，或根本没有），因而不可能带派生扩展离开。"不可判定" = 有完整的结构化导出或直接存储访问，可以写 .mem 适配器，但没有现成的第三方格式路径，也没有测量。"待 E6" = 由我们的 E6 测量。
- 标记：† = 推断（inferred），‡ = 按构造成立、未测量；无标记 = 有文档、代码或论文直接支持，或待我们的实验。

## 表 1：§3.4 检验表（计划列出的 16 类系统 + 1 个拆分行）

| 系统 | 拥有者 | D | B | T1 换写入模型 | T2 离开 | T3 效用下限 | T4 成本上限 |
|---|---|---|---|---|---|---|---|
| ChatGPT memory | 厂商 | 0† | 0† | 不可判定：写入模型不可控不可见† | 不通过：无记忆导出，只能让模型复述 | 不可判定：黑盒† | 不可判定：成本不公开 |
| Claude memory | 厂商 | 0† | 0 | 不可判定：写入模型未公开† | 不通过：散文复述导出 | 不可判定：黑盒† | 不可判定：成本不公开† |
| Claude API memory tool | agent | 0† | 0† | 不可判定：未测† | 不可判定：有 .mem 摄入，无派生字段 | 不可判定：无 oracle 对照 | 不可判定 |
| Gemini memory | 厂商 | 0† | 0† | 不可判定：写入模型未公开† | 不通过：记忆不可导出 | 不可判定：黑盒† | 不可判定：成本不公开† |
| Apple semantic index | 平台 | 部分† | 0† | 不可判定：模型不可控† | 不通过：只进不出 | 不可判定：黑盒† | 不可判定† |
| Mem0 (self-hosted) | agent | 0 | 0 | 不可判定：无 θ_w 键、不重派生，保留率 1 是未更新而非等效‡ | 不可判定：E6 无 Mem0 臂 | 不通过：E5 对 oracle −16.9 pp [−20.6, −13.2]（15 个短世界） | 通过：E5 服务成本 1.14% 全量重读 |
| Graphiti / Zep | agent | 部分 | 仅时间有效性 | 不可判定：E5 未跑（预算，§12；实测每事件 6.1 次调用） | 不可判定：E6 无 Graphiti 导入臂 | 不可判定：E5 未跑 | 不可判定：E5 未跑 |
| Letta (.af) | agent | 0 | 0 | 不可判定：未测† | 不可判定：只有自有格式 | 不可判定 | 不可判定 |
| Cognee | agent | 部分 | 0 | 不可判定：未测 | 不可判定：自有 COGX，可导入他家 | 不可判定 | 不可判定 |
| Honcho | agent | 部分 | 0 | 不可判定：未测† | 不可判定：只有列表 API | 不可判定：厂商自报 | 不可判定：厂商自报 |
| Supermemory | 厂商 | 部分† | 仅时间有效性† | 不可判定 | 不可判定：只有列表 API | 不可判定：厂商自报 | 不可判定：厂商自报 |
| Khoj | 人 | 0 | 0 | 不可判定：未测 | 不可判定：只有列表 API | 不可判定† | 不可判定† |
| Opal | 厂商 | 部分 | 0 | 不可判定：无代码 | 不可判定：未讨论 | 不可判定：无 oracle 对照 | 不可判定 |
| Plain RAG | 人 | 不适用 | 不适用 | 通过（构造上，无 θ_w）‡ | 通过（构造上，无派生）‡ | 不通过：E3 对 oracle −14.3 / −22.1 / −17.5 pp | 通过：E3 服务成本 ≤ 1.8% 全量重读 |
| Solid pod | 人 | 不适用 | 不适用 | 通过（构造上）‡ | 通过（构造上；无标准迁移工具）‡ | 不通过：无读取路径† | 不可判定：无服务路径† |
| User as Code | agent | 部分 | 0 | 不可判定：只部分重派生 | 不可判定：只有目录复制 | 不可判定：论文数字提示可能通过 | 不可判定：论文数字提示可能通过 |
| 零系统（ours） | 人 | 不适用 | 不适用 | 通过（构造上，E2 不测）‡ | 通过（构造上，无派生状态）‡ | 不通过：E3 对 oracle −3.6 / −18.1 / −13.6 pp | 通过：E3 服务成本 ≤ 1.8% 全量重读 |
| HoldMem (1,1) | 人 | 1 | 1 | 通过：E2 三档 TOST ±3 pp（平级 −1.36 pp [−1.86, −0.86]）；自由文本项例外（探索性） | 通过：E6 带扩展导入三项差均为 0（导入方为第二个 HoldMem 实现）；不带扩展漏失效 +100 pp | 通过：E3 对 oracle +0.7 / +1.3 / +12.2 pp | 通过：E3 服务成本 ≤ 2.1% 全量重读 |
| HoldMem (0,1) | 人 | 0 | 1 | 不可判定：T1 只对 (1,1) 预注册；保留率 0.984 / 0.996 / 0.970 仅描述 | 不可判定：E6 只导出 (1,1) | 不可判定：E3 只测 (1,1) | 不可判定：E3 只测 (1,1) |
| HoldMem (1,0) | 人 | 1 | 0 | 不可判定：B=0 不重派生，保留率 1.0 是未更新而非等效‡ | 不可判定：E6 只导出 (1,1) | 不可判定：E3 只测 (1,1) | 不可判定：E3 只测 (1,1) |
| HoldMem (0,0) | 人 | 0 | 0 | 不可判定：B=0 不重派生，保留率 1.0 是未更新而非等效‡ | 不可判定：E6 只导出 (1,1) | 不可判定：E3 只测 (1,1) | 不可判定：E3 只测 (1,1) |

## 表 2：起点审计 `reference/research_2/system_audit.json` 中的额外系统

起点审计用的是 S/O/Q/U 四项标准，不是 T1–T4；这里按本文件口径重判。其中 Graphiti、Cognee、Mem0、Solid 已并入表 1。

| 系统 | 拥有者 | D | B | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|---|---|
| Hindsight | agent | 部分 | 0 | 不可判定 | 不可判定：自有格式 | 不可判定† | 不可判定† |
| OpenViking | agent | 部分 | 0 | 不可判定 | 不可判定：自有格式 | 不可判定† | 不可判定† |
| MemOS | agent | 部分 | 0 | 不可判定 | 不可判定：自有格式 | 不可判定† | 不可判定† |
| GraphRAG | agent | 部分 | 0 | 不可判定† | 不可判定 | 不可判定† | 不可判定† |
| MemTX | agent | 部分 | 0 | 不可判定 | 不可判定 | 不可判定 | 不可判定 |
| Document-to-Database | agent | 部分 | 0 | 不可判定 | 不可判定 | 不可判定 | 不可判定 |
| AnnoIndex | agent | 部分 | 0† | 不可判定 | 不可判定 | 不可判定 | 不可判定 |
| Agentic Context Cracking | agent | 部分 | 0 | 不可判定 | 不可判定 | 不可判定 | 不可判定 |
| DuckDB / pgvector / LOTUS | 不适用（组件） | 不适用 | 不适用 | 不可判定：组件 | 不可判定：组件 | 不可判定：组件 | 不可判定：组件 |

## 计数

表 1（21 行）：

| 测试 | 通过 | 不通过 | 不可判定 | 待 E2 | 待 E3 | 待 E5 | 待 E6 |
|---|---|---|---|---|---|---|---|
| T1 | 4 | 0 | 17 | 0 | 0 | 0 | 0 |
| T2 | 4 | 4 | 13 | 0 | 0 | 0 | 0 |
| T3 | 1 | 4 | 16 | 0 | 0 | 0 | 0 |
| T4 | 4 | 0 | 17 | 0 | 0 | 0 | 0 |

表 1 + 表 2（30 行）：

| 测试 | 通过 | 不通过 | 不可判定 | 待 E2 | 待 E3 | 待 E5 | 待 E6 |
|---|---|---|---|---|---|---|---|
| T1 | 4 | 0 | 26 | 0 | 0 | 0 | 0 |
| T2 | 4 | 4 | 22 | 0 | 0 | 0 | 0 |
| T3 | 1 | 4 | 25 | 0 | 0 | 0 | 0 |
| T4 | 4 | 0 | 26 | 0 | 0 | 0 | 0 |

表 1 中测量得到的"通过"：HoldMem (1,1) 的 T1、HoldMem (1,1) 的 T2、HoldMem (1,1) 的 T3、HoldMem (1,1) 的 T4、Mem0 (self-hosted) 的 T4、Plain RAG 的 T4、零系统（ours） 的 T4；按构造成立的：Plain RAG 的 T1、Plain RAG 的 T2、Solid pod 的 T1、Solid pod 的 T2、零系统（ours） 的 T1、零系统（ours） 的 T2。外部系统里测量得到的"通过"只有 Mem0 的 T4（成本上限对任何预算受限的检索式系统都不构成约束）。T2 的 4 个"不通过"（ChatGPT、Claude、Gemini、Apple）都来自厂商或平台自己的文档；T3 的 1 个"不通过"（Solid）是推断。

## Portable Memory (.mem) 与 T2

- **格式现状**：`portable-memory` 0.3.0（PyPI，2026-09-14），格式 1.1.0，MIT；自述是开放格式提案，不是定稿标准 [PM-REPO, PM-PYPI]。
- **适配器**：Python 与 Swift 两个 SDK 都只带四个**单向摄入**适配器（外部 → .mem）：Mem0Adapter（Mem0 导出 JSON，只映射为 episode，图关系不映射）、OpenAIAdapter（ChatGPT `conversations.json` 或 saved-memory 列表）、ClaudeAdapter（Claude Code / API memory tool 的 markdown 文件，不是 claude.ai 消费者记忆）、TransferTextAdapter（ChatGPT / Claude / Gemini 之间粘贴的 `[date] - memory` 文本）。出口只有 `mem render`：把 .mem 渲染成可粘贴的文本 [PM-ADAPTERS, PM-MEM0, PM-SWIFT]。规范的映射表列了 Letta，但没有 Letta 适配器；Zep 只出现在"先前工作"里；唯一另提的 Engram 适配器 PR 未合并 [PM-SPEC-MAP, PM-PR8]。
- **没有任何记忆引擎原生读写 .mem**：在 Mem0、Graphiti、Letta、Cognee、Honcho、Supermemory、Khoj、MemOS、Hindsight 的仓库里做代码搜索与 grep，均为 0 命中 [GH-SEARCH]。所以"由另一实现按其自身导入路径导入"目前只有两条路：我们自己的第二个 HoldMem 实现，以及 Mem0 的 `add(infer=False)` 原文写入（`import_to_mem0`；SDK 没有 .mem → Mem0 的反向适配器）[CODE-MEMEXPORT]。
- **派生扩展能放在哪里**：格式 1.1 只保证 episode 记录上的 `ext` 与未知记录 kind 原样往返；fact / edge 上的未知字段会被两个参考读取器丢弃（RFC-0002 拟在 1.2 修正，评论期到 2026-10-05）[PM-SPEC-EXT]。本仓库已实测：fact 上的 `ext` 在导入时被丢弃，自定义 kind 与 episode `ext` 存活 [CODE-EXTSURV]。格式自带的来源只到 episode 粒度（fact.episodeID、edge.evidenceEpisodeIDs），没有定义版本与写入模型字段；"断言模型"要到 1.2 的 RFC-0003 才有提案 [PM-SPEC-PROV, PM-RFC3]。
- **E6 当前覆盖**：驱动只导出 HoldMem (1,1)，只导入到第二个 HoldMem，没有 Mem0 / Graphiti 臂，也没有零系统或其余三格的臂 [CODE-E6]。所以表 1 中 Mem0、Graphiti 与 HoldMem 其余三格的 T2 记为不可判定（未测）。

## 与计划 §3.4 预期的偏差

| 系统 | 计划预期 | 本审计 |
|---|---|---|
| Claude API memory tool | （计划未单列） | 新增行 |
| Apple semantic index | D1（结构化实体）B0; T2 不通过（只进不出） | D 由 1 改为部分 |
| Mem0 (self-hosted) | D0 B0（v3 只 ADD）; T1/T3/T4 待 E5; T2 待 E6（有 .mem 适配器） | E5 按 §12 缩为 15 个 3 个月世界（decisions.md 2026-09-29）；T1 不适用（B=0 不重派生）；T2 不可判定（E6 无 Mem0 臂） |
| Graphiti / Zep | D 部分; B 时间有效性; T1/T3/T4 待 E5; T2 待 E6 | T2 待 E6 的前提当前不成立 |
| Letta (.af) | D0 B0; T1 待审计; T2 部分（自有格式） | T2 记为不可判定（自有格式存在，第三方格式路径不存在） |
| Cognee | D 部分; B0; T1/T2 待审计 | B 形式上为 0，但有源级级联（与计划一致，补充说明） |
| Honcho | D0 B0; T1/T2 待审计 | D 记为部分（有内部源引用） |
| Supermemory | D0 B0; T1/T2 待审计 | B 记为仅时间有效性（版本链），D 记为部分 |
| Opal | 人; D 部分（KG）; B0 | 拥有者由人改为厂商 |
| Plain RAG | T1 通过; T2 通过; T3 待 E3; T4 不通过（预期） | T4 的"不通过"预期与 prereg 预算不一致（推断） |
| User as Code | 人; D1 B0; T1 通过; T2 部分 | D 由 1 改为部分；T1 由通过改为不可判定；T2 由部分改为不可判定 |
| 零系统（ours） | T1 通过; T2 通过; T3/T4 待 E3 | T2 由通过改为待 E6（无导出路径与 E6 臂） |

## 已观测、推断与待实验

**已观测**（有文档、代码或论文直接支持）：厂商三家都没有带来源的记忆条目，记忆可以不随源删除而删除（ChatGPT 的 saved memory、Claude 的条目；Gemini 相反，靠删除源对话来删除），且没有结构化记忆导出，所以 T2 不通过。Mem0 OSS 的 v3 管线只 ADD，不记写入模型，原始消息只留 10 条。Graphiti 的失效只由新的矛盾信息触发。Letta、Honcho、Khoj 由 LLM 原地改写或删除，没有键。Cognee 与 Hindsight 有源级级联，但键里没有定义与写入模型。审计的外部系统没有一个记录逐项写入模型。.mem 生态只有单向摄入适配器；fact 上的 ext 在导入时被丢弃。

**推断**（标 †）：各厂商的 D=0（条目内部表示不公开，按可见形态判断）。Apple 条目的 D=部分。Supermemory 的 B（服务端不公开，依据客户端 schema）。Solid 的 T3 不通过。Plain RAG 与零系统的 T4 很可能不受约束：按 prereg，读者证据区为 4,096 token，而全量重读约 2.2M token，任何预算受限的检索系统的服务成本都远低于 0.5×。若要 T4 有区分力，可考虑把摊销的摄入与维护成本计入，这一改动需要由人决定。

**待我们的实验**：无。E2、E3、E5、E6 的结果已填入（E5 只跑了 Mem0，15 个 3 个月世界；Graphiti 因实测成本未跑，记不可判定）；Mem0、Graphiti、零系统与 HoldMem 其余三格的 T2 因 E6 没有对应的导入臂而记为不可判定，Mem0 的 T2 预测为 E6"不带扩展"的结果（未测）。

**概念上的注意点**：T1 的陈述预设 θ_w 进键并全部重派生。B=0 的系统（包括 HoldMem 的两个 B=0 格、Mem0、Letta 等）不满足这个前提：换写入模型时它们什么都不重派生，存活率约为 1，但这不是"重派生后等效"。E2 报告这些格时应单独说明，不宜把它们的存活率写成 T1 通过。

## 各系统说明与证据

### ChatGPT memory

*ChatGPT memory (saved memories + 2026 'dreaming' memory summary)* — 拥有者：厂商（OpenAI-hosted; user can view/edit/delete, not operate）

2026-06-04 起默认以后台合成的记忆摘要（dreaming）替代逐条列表，可切回 saved memories；摘要自述可能遗漏细节。来源只在回答级别给出（Sources），不在记忆条目上。删除对话不必然删除由它派生的 saved memory。数据导出只列对话与账号数据，跨账号迁移明确不迁移记忆，也没有导入功能；唯一出口是让模型用散文复述记忆，再粘贴出去（MacPaw 的 TransferTextAdapter 可把这段文字收进 .mem，但没有任何派生字段可带）。OpenAI 页面直接抓取返回 403，经 r.jina.ai 代理读取全文。

- D = 0（inferred）：Memory is a background-synthesized prose summary plus optional free-text saved memories; attribution ('Sources') is per response, not per item, and has no spans.
- B = 0（documented + inferred；实际键分量：none documented (per-response Sources only)）：No key: an LLM process rewrites memory in the background and deleting a source chat does not necessarily delete saved memories derived from it; writer model and per-item source are not documented.
- T1 = 不可判定（inferred）：Writer (dreaming) model is vendor-chosen and undocumented; no per-item theta_w; users cannot trigger re-derivation; S_wr cannot be applied or observed from outside.
- T2 = 不通过（documented）：No memory export: the data export covers chats and account data, memories do not transfer between accounts, no import exists; the only exit is asking the model to print memories as prose (-> .mem only through TransferTextAdapter, no derivation fields possible).
- T3 = 不可判定（inferred）：A controlled world cannot be ingested and no oracle comparison is possible from outside; vendor evals compare memory versions only.
- T4 = 不可判定（documented）：Serving cost is not exposed; the ~5x compute reduction is relative to the earlier dreaming system, not to a full re-read.
- 证据：[OAI-DREAM] https://openai.com/index/chatgpt-memory-dreaming/；[OAI-FAQ] https://help.openai.com/en/articles/8590148-memory-faq；[OAI-RN] https://help.openai.com/en/articles/6825453-chatgpt-release-notes；[OAI-EXPORT] https://help.openai.com/en/articles/7260999-how-do-i-export-my-chatgpt-history-and-data；[OAI-TRANSFER] https://help.openai.com/en/articles/9106926-transfer-exported-conversations-between-chatgpt-accounts；[PM-ADAPTERS] portable_memory/adapters/{__init__,mem0,openai,claude,transfer}.py; portable_memory/cli.py @f341f0703d739edfdf94eba58e66a4cc7c925df7；[BG-CARDS] docs/background/IDEA_PAIN_POINTS_AND_CARDS_ZH.md §1.1 I1, I2, I5, I8, I9; §1.2 L9, L12 @56876e3634ca9493d9511e67d98e935be81cb887

### Claude memory

*Claude memory (claude.ai consumer, incl. import/export)* — 拥有者：厂商（Anthropic-hosted; per-project memory spaces）

2026-07-10 起记忆改为对话中逐条保存的分类条目（取代每日摘要），8 月起按 Topics 展示。帮助页明确写：对话删除或过期后，由它生成的记忆条目不会被删除。导入 = 把别家助手按 `[date saved] - memory content` 格式吐出的文本粘贴进来，由 Claude 抽取，标注实验性、可能不完整；导出 = 让 Claude 逐字写出记忆再复制。消费者数据导出不列记忆；Team/Enterprise 导出包含记忆条目但格式未公开（未计入判定）。

- D = 0（inferred）：Entries are prose under topic headings; no per-entry source reference or typed value is documented.
- B = 0（documented；实际键分量：none documented）：Claude updates entries during conversations (LLM-judged); when a conversation is deleted or expires, memory entries generated from it are not removed; no key.
- T1 = 不可判定（inferred）：Which model writes memory entries is undocumented; no per-entry theta_w; no user-triggered re-derivation.
- T2 = 不通过（documented）：Export = ask Claude to write its memories out verbatim and copy-paste (model-generated prose, completeness not guaranteed); consumer data export does not list memory; import is an experimental LLM re-extraction of pasted text. No derivation extension is possible. (Team/Enterprise exports include memory entries; format undocumented.)
- T3 = 不可判定（inferred）：Black box; no published comparison.
- T4 = 不可判定（inferred）：Serving cost not exposed.
- 证据：[ANT-IMPEXP] https://support.claude.com/en/articles/12123587-import-and-export-your-memory-from-claude；[ANT-MEM] https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context；[ANT-RN] https://support.claude.com/en/articles/12138966-release-notes；[ANT-EXPORT] https://support.claude.com/en/articles/9450526-export-your-claude-data；[ANT-IMPORT-PAGE] https://claude.com/import-memory；[PM-ADAPTERS] portable_memory/adapters/{__init__,mem0,openai,claude,transfer}.py; portable_memory/cli.py @f341f0703d739edfdf94eba58e66a4cc7c925df7；[BG-CARDS] docs/background/IDEA_PAIN_POINTS_AND_CARDS_ZH.md §1.1 I1, I2, I5, I8, I9; §1.2 L9, L12 @56876e3634ca9493d9511e67d98e935be81cb887

### Claude API memory tool

*Claude API memory tool (developer-managed /memories files)* — 拥有者：agent（client-side files the developer stores; extra row split from 'Claude memory'）

与消费者 Claude memory 不同：文件存在开发者一侧，由开发者决定存储。协议只有文件操作命令，没有来源、模型或版本字段。MacPaw 的 ClaudeAdapter 读取的正是这种目录（及 Claude Code auto-memory），不是 claude.ai 的消费者记忆。

- D = 0（inferred）：Free-form files edited by the model; the tool protocol has no provenance, model or version fields (a developer could add structure).
- B = 0（inferred；实际键分量：none）：The model edits files through str_replace/delete/rename at its own judgment; nothing records a key or invalidates on change.
- T1 = 不可判定（inferred）：Model is selectable per request but files record no theta_w and nothing re-derives; runnable in principle, not in E2/E5.
- T2 = 不可判定（code）：Developer owns the complete files and Portable Memory's ClaudeAdapter ingests the directory into .mem episodes, but there are no derivation fields to carry and no import measurement.
- T3 = 不可判定（documented）：Published +39% is against a no-memory baseline, not an oracle.
- T4 = 不可判定（documented）：The 84% token reduction belongs to context editing, not the memory tool.
- 证据：[ANT-MEMTOOL] https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool；[ANT-CTX] https://www.anthropic.com/news/context-management；[PM-ADAPTERS] portable_memory/adapters/{__init__,mem0,openai,claude,transfer}.py; portable_memory/cli.py @f341f0703d739edfdf94eba58e66a4cc7c925df7

### Gemini memory

*Gemini memory / saved info ('Instructions for Gemini'), Personal Intelligence* — 拥有者：厂商（Google-hosted）

"记忆"来自过往对话（需开启 Keep Activity），用来生成个性化 insight；没有可查看的条目列表。要删除 Gemini 记住的东西，文档要求删除所有含该信息的对话，删除后有短暂延迟——这是按源删除传播，但不是键。"Saved info" 现称 Instructions，是用户自己写的文本。导入：粘贴提示词输出或上传聊天 ZIP；Takeout 只列 Gems 与活动记录。**不确定性最高的格之一**：若记忆实际是查询时对过往对话的检索而无持久派生项，则 D、B 应为不适用。

- D = 0（inferred (low confidence)）：Derived 'insights' are not exposed as items; the visible Instructions are user-authored free text. If memory is query-time retrieval over past chats, D would be N/A.
- B = 0（documented + inferred；实际键分量：source-chat deletion only (documented), no revision/definition/model key）：To delete something Gemini remembered, the user deletes all chats containing it (short delay): deletion propagates from sources, but no key, no item list, no definition or model version is documented.
- T1 = 不可判定（inferred）：Memory works with all models in the picker; which model derives insights is undocumented; no re-derivation control.
- T2 = 不通过（documented）：No memory export: Takeout lists Gems and Gemini Apps activity, not memory or Instructions; import only by pasting a prompt dump or uploading a chat ZIP.
- T3 = 不可判定（inferred）：Black box.
- T4 = 不可判定（inferred）：Serving cost not exposed.
- 证据：[GOO-MEM] https://support.google.com/gemini/answer/16598469；[GOO-PERS] https://support.google.com/gemini/answer/15637730；[GOO-HUB] https://support.google.com/gemini/answer/13594961；[GOO-INSTR] https://support.google.com/gemini/answer/16598625；[GOO-SWITCH] https://blog.google/innovation-and-ai/products/gemini-app/switch-to-gemini-app/；[GOO-IMPORT] https://support.google.com/gemini/answer/16868299; https://gemini.google/import-memory/；[GOO-TAKEOUT] https://support.google.com/gemini/answer/16920332；[GOO-PI] https://blog.google/innovation-and-ai/products/gemini-app/personal-intelligence/；[BG-CARDS] docs/background/IDEA_PAIN_POINTS_AND_CARDS_ZH.md §1.1 I1, I2, I5, I8, I9; §1.2 L9, L12 @56876e3634ca9493d9511e67d98e935be81cb887

### Apple semantic index

*Apple semantic index / Apple Intelligence personal context (iOS 27, WWDC26)* — 拥有者：平台（on-device index, accessible to the system alone）

第三方 app 通过 IndexedEntity / App Schemas 把类型化实体捐给系统语义索引（按实体 id upsert，可删除，系统可要求重建索引）；检索只限本 app 沙箱，索引内容"只对系统可用"。条目是 app 数据的类型化副本而非 LLM 派生，所以 D 记为"部分"而不是计划预期的 1：值可追到源对象，但没有片段级复核，而 Siri 在请求时派生的东西不可见。用户侧没有导出。

- D = 部分（inferred）：Entries are app-donated typed App Schema entities with source identifiers (copies of app data, not LLM-derived); no span or definition check; whatever Siri derives at request time is not exposed.
- B = 0（inferred；实际键分量：source object id (app-driven upsert/delete); no revision, definition or model key）：Freshness depends on the app re-donating (upsert by entity id), delete APIs, and system reindex after OS updates or data changes; no key invalidates items.
- T1 = 不可判定（inferred）：Neither users nor developers control the model; full reindex after OS updates is documented but not tied to a model version.
- T2 = 不通过（documented）：The index is local and available to the system alone; third-party apps can retrieve only their own content; no user export or import of the index or personal context exists (only the PCC request report).
- T3 = 不可判定（inferred）：Black box.
- T4 = 不可判定（inferred）：Not published.
- 证据：[APL-240] https://developer.apple.com/videos/play/wwdc2026/240/；[APL-GUIDE] https://developer.apple.com/wwdc26/guides/apple-intelligence/; https://developer.apple.com/videos/play/wwdc2026/102/；[APL-SPOT] https://developer.apple.com/documentation/appintents/making-app-entities-available-in-spotlight；[APL-CSS] https://developer.apple.com/documentation/corespotlight/cssearchableindex；[APL-8011] https://developer.apple.com/videos/play/wwdc2026/8011/；[APL-SUPPORT] https://support.apple.com/en-us/102321；[APL-PCC] https://www.apple.com/legal/privacy/data/en/intelligence-engine/；[BG-CARDS] docs/background/IDEA_PAIN_POINTS_AND_CARDS_ZH.md §1.1 I1, I2, I5, I8, I9; §1.2 L9, L12 @56876e3634ca9493d9511e67d98e935be81cb887

### Mem0 (self-hosted)

*Mem0 open source, self-hosted (v3 ADD-only pipeline, mem0ai 2.2.x)* — 拥有者：agent（developer-run store scoped by user_id/agent_id/run_id）

"v3"指记忆管线而非包版本：mem0ai 2.0.0 起默认单次 ADD-only 抽取，信息变化时新事实与旧事实并存；issue #4956（陈旧/矛盾事实）仍开放，软替代 PR 未合并；#5330（衰减）以插件替代关闭。记录只有文本、hash、时间戳与作用域 id，无写入模型、无源引用；原始消息只留最近 10 条（storage.py L279-286），所以不另存原文就无法全量重派生——本项目 native.py 以 keep_raw 解决。OSS 没有导出；MacPaw 的 Mem0Adapter 只把 Mem0 导出单向收进 .mem（只映射为 episode）。厂商自己的 State of AI Agent Memory 2026 把高相关记忆的陈旧列为未解决问题。

- D = 0（code）：The memory value is free text ('data') plus an MD5 hash; no source reference, span or typed value. E5: structured repair precision 1.00 (n=289), descriptive only (kappa_D below the bar); carryover inside the HoldMem (0,0) prediction on 5 of 6 pre-registered measures.
- B = 0（code + documented；实际键分量：none (scope ids, attributed_to, timestamps only)）：The v3 pipeline is ADD-only (one extraction call, inserts only; new facts stored alongside old ones); UPDATE/DELETE only through explicit API calls; supersede ('Dream') and decay are Platform-only. E5 orphan probe: persistence 0.549 [0.490, 0.609] -> B=0 by the 0.5 threshold, between the white-box B=0 (0.90) and B=1 (0.13-0.17) cells.
- T1 = 不可判定（by construction）：No per-item theta_w and no re-derivation path: a writer switch changes no stored memory, so retention 1 would mean nothing was updated, not equivalence (as for HoldMem (0,0)); S_wr was not run on Mem0.
- T2 = 不可判定（not tested）：OSS has no export ('Memory Export: Not available'); the only .mem route is get_all JSON -> Portable Memory Mem0Adapter (episodes only); E6 has no Mem0 arm. Prediction (not measured): the no-extension outcome of E6, since Mem0 has no binding that would read a derivation record.
- T3 = 不通过（measured (E5, 15 short worlds)）：Static ContractValid 0.660 vs oracle context 0.829 on the same 15 three-month worlds: -16.9 pp [-20.6, -13.2] (floor -3 pp); zero system -16.8, RAG -20.7 pp on the same worlds.
- T4 = 通过（measured (E5, 15 short worlds)）：Reader tokens per query 1.14% of one full re-read at t0 (ceiling 50%); ingestion: 1.0 LLM call and 10.4k prompt tokens per event (not part of T4).
- 证据：[M0-CODE] https://github.com/mem0ai/mem0 @94c3fe9f238f3dbf29c9ce98643bd71eb13077cd: mem0/memory/main.py L881-1220 (V3 phased batch pipeline, inserts only, history event ADD), L1030-1039 (payload: data, hash, created_at, updated_at, attributed_to, scope ids, metadata); mem0/configs/prompts.py L472; mem0/memory/storage.py L108-118 (history table), L279-286 (evict messages beyond 10 per scope)；[M0-MIG] https://docs.mem0.ai/migration/oss-v2-to-v3 (repo docs/migration/oss-v2-to-v3.mdx L16, L38, L309)；[M0-PVO] docs/platform/platform-vs-oss.mdx L73, L81 @94c3fe9f238f3dbf29c9ce98643bd71eb13077cd；[M0-4956] https://github.com/mem0ai/mem0/issues/4956；[M0-5330] https://github.com/mem0ai/mem0/issues/5330；[M0-STATE] https://mem0.ai/blog/state-of-ai-agent-memory-2026；[M0-PAPER] arXiv:2504.19413 §4.2 Table 2；[CODE-NATIVE] src/memshift/adapters/native.py module docstring L1-35 @56876e3634ca9493d9511e67d98e935be81cb887；[CODE-MEMEXPORT] src/memshift/adapters/mem_export.py module docstring; export_holdmem L244; import_to_holdmem L444; import_to_mem0 L529 @56876e3634ca9493d9511e67d98e935be81cb887；[CODE-E6] src/memshift/experiments/e6.py module docstring @56876e3634ca9493d9511e67d98e935be81cb887；[PM-MEM0] portable_memory/adapters/mem0.py module docstring @f341f0703d739edfdf94eba58e66a4cc7c925df7；[PM-ADAPTERS] portable_memory/adapters/{__init__,mem0,openai,claude,transfer}.py; portable_memory/cli.py @f341f0703d739edfdf94eba58e66a4cc7c925df7；[E27] reference/research_2/evidence_registry.jsonl#E27; reference/research_1/evidence/sources.json#S-MEM0；[BG-CARDS] docs/background/IDEA_PAIN_POINTS_AND_CARDS_ZH.md §1.1 I1, I2, I5, I8, I9; §1.2 L9, L12 @56876e3634ca9493d9511e67d98e935be81cb887；[RUN-STAGE3-E5] runs/stage3_e5/analysis/e5_natives.json; runs/stage3_e5_cost/analysis/e5_cost.json; prereg/decisions.md 2026-09-29 (E5 reduction; T3 reference)

### Graphiti / Zep

*Graphiti (open source) / Zep (hosted)* — 拥有者：agent（developer library partitioned by group_id; Zep Cloud = developer project）

EntityEdge 带 fact、episodes、valid_at / invalid_at / expired_at；失效只在 LLM 判定新信息与旧边矛盾时发生（edge_operations.py L538-573）。没有"修订 episode"的路径；删除创建边的 episode 会删边，但它曾使之失效的边不会恢复（Zep 文档同样写明）；本体变更不回溯。每条边不记模型。没有导出 API；.mem 的 edge 结构几乎一一对应 Graphiti，但没有适配器。

- D = 部分（code）：Typed edges (relation name, fact text, Pydantic attributes) with episode-UUID provenance; no spans and no deterministic checker.
- B = 仅时间有效性（code + paper + documented；实际键分量：valid_at/invalid_at/expired_at + episode UUIDs; no source revision, definition or model key）：Edges are invalidated only when an LLM judges that new information contradicts them; a revised episode is not re-derived (uuid reuse reprocesses stored content); deleting the creator episode removes its edges but does not restore edges it invalidated; ontology changes are not re-applied.
- T1 = 待 E5（pending experiment）：Graphiti: LLM switchable, not recorded per edge, no re-derivation. Zep Cloud: BYOM exists, recording/reprocessing undocumented (undeterminable).
- T2 = 待 E6（pending experiment）：No export API (Graphiti: raw graph DB; Zep: paginated list reads) and no .mem adapter, although Graphiti edges map field-for-field onto .mem edges; the current E6 driver has no Graphiti arm.
- T3 = 待 E5（pending experiment）：Prior: vendor paper reports Zep above full context on LongMemEval (71.2% vs 60.2%, gpt-4o).
- T4 = 待 E5（pending experiment）：Prior: ~1.6k vs ~115k tokens per query (vendor).
- 证据：[GR-CODE] https://github.com/getzep/graphiti @6b4b56ff6f4b1e4e69c3c3c5487cf1b8762c483a: graphiti_core/edges.py L263-285 (EntityEdge: fact, episodes, valid_at, invalid_at, expired_at, attributes); graphiti_core/utils/maintenance/edge_operations.py L538-573 (resolve_edge_contradictions), L576-620 (LLM timestamps), L623-847 (resolve_extracted_edge); graphiti_core/graphiti.py L1156-1168 (uuid reuse loads stored episode), L1824-1852 (remove_episode); graphiti_core/llm_client/config.py L40-66；[ZEP-PAPER] arXiv:2501.13956 §2.2.3, §4.3.2 Table 2；[ZEP-DOCS] https://help.getzep.com/source-traceability; https://help.getzep.com/deleting-data-from-the-graph; https://help.getzep.com/customizing-graph-structure; https://help.getzep.com/reading-data-from-the-graph; https://help.getzep.com/bring-your-own-llm；[CODE-NATIVE] src/memshift/adapters/native.py module docstring L1-35 @56876e3634ca9493d9511e67d98e935be81cb887；[CODE-E6] src/memshift/experiments/e6.py module docstring @56876e3634ca9493d9511e67d98e935be81cb887；[PM-SPEC-PROV] Spec/portable-memory-spec.md §2 'Identity, time & provenance' @f341f0703d739edfdf94eba58e66a4cc7c925df7；[PM-SPEC-MAP] Spec/portable-memory-spec.md §10 'Mapping reference', §12 'Prior art' @f341f0703d739edfdf94eba58e66a4cc7c925df7；[GH-SEARCH] GitHub code search 'portable-memory repo:<r>' for mem0ai/mem0, getzep/graphiti, letta-ai/letta, topoteretes/cognee, plastic-labs/honcho, supermemoryai/supermemory, khoj-ai/khoj, MemTensor/MemOS, vectorize-io/hindsight; plus grep of the cloned repos；[E20] reference/research_2/evidence_registry.jsonl#E20; reference/research_1/evidence/sources.json#S-GRAPHITI

### Letta (.af)

*Letta (Letta Code MemFS; legacy Agent File .af)* — 拥有者：agent（person when Letta Code runs with local MemFS）

letta 主仓库已成落地页，现行代码在 letta-code（MemFS：git 管理的 markdown 记忆）；V1 服务器归档且官方声明不作研究依据。.af 文档归在 "V1 SDK (legacy)"，不含 archival passages 与 block 历史，README 自述其他框架"理论上"可加载、跨框架转换器未完成。反思子代理按提示词原地改写陈旧条目。Cognee 有 Letta 导入器（非 .mem）。

- D = 0（code）：Memory is free-text markdown (frontmatter name/description only) or block values; no source references.
- B = 0（code + documented；实际键分量：none (git history of edits, commit author = agent id)）：The agent or its reflection subagent rewrites stale entries in place at LLM judgment; git records edits but there is no key and no invalidation on source, definition or model change.
- T1 = 不可判定（code + inferred）：Model switchable (subagents inherit it) but not recorded on memory and nothing re-derives; not in E5.
- T2 = 不可判定（documented + code）：Own formats only: `letta memory export` copies the MemFS directory; .af is legacy V1, omits archival passages and block history, and no other framework imports it; no .mem adapter (the .mem spec maps Letta but ships none).
- T3 = 不可判定（documented）：Only a LoCoMo 74% vendor figure without a full-context floor.
- T4 = 不可判定（documented）：Sleep-time compute numbers concern reasoning compute, not serving vs full re-read.
- 证据：[LT-REPO] https://github.com/letta-ai/letta @5bcdd177d70fa2b31a754cfcd801e77b2e1ab16a README.md, AGENTS.md；[LT-CODE] https://github.com/letta-ai/letta-code @c864f1532b328aab4bb76cc68a86d5f014de27b8: src/memory-frontmatter.ts L64-66; src/agent/memory-git.ts L1184-1206; src/agent/subagents/builtin/reflection-v2.md L5, L102; src/cli/subcommands/memory.ts L284-316；[LT-MEMFS] https://docs.letta.com/concepts/memfs; https://docs.letta.com/configuration/memory；[AF-REPO] https://github.com/letta-ai/agent-file @78212eb571e59e10b35b924375a997319b89c280 README.md L170-199, L191, L230-236；[LT-AFDOC] https://docs.letta.com/guides/core-concepts/agent-file；[PM-SPEC-MAP] Spec/portable-memory-spec.md §10 'Mapping reference', §12 'Prior art' @f341f0703d739edfdf94eba58e66a4cc7c925df7；[CG-CODE] https://github.com/topoteretes/cognee @c4cd8ceb9509dff6bddfabdadbeab7cc040bc32b: cognee/infrastructure/engine/models/DataPoint.py; cognee/infrastructure/databases/provenance/source_refs.py; cognee/infrastructure/databases/unified/provenance_delete_planner.py; cognee/api/v1/update/update.py, incremental.py; cognee/api/v1/forget/forget.py; cognee/modules/pipelines/operations/run_tasks_data_item.py ~L197-207; cognee/modules/pipelines/models/PipelineRun.py; cognee/modules/cognify/config.py L22; cognee/api/v1/export/export.py; cognee/modules/migration/cogx.py; cognee/modules/migration/sources/{mem0,zep,letta,langmem,cogx_archive}.py；[GH-SEARCH] GitHub code search 'portable-memory repo:<r>' for mem0ai/mem0, getzep/graphiti, letta-ai/letta, topoteretes/cognee, plastic-labs/honcho, supermemoryai/supermemory, khoj-ai/khoj, MemTensor/MemOS, vectorize-io/hindsight; plus grep of the cloned repos

### Cognee

*Cognee (open source; Cognee Cloud)* — 拥有者：agent（developer library keyed by user/dataset; person if self-hosted by an individual; vendor on Cognee Cloud）

源删除按引用计数级联到派生节点与边；调用 update() 时只对变化的分块重抽取。这是外部引擎里最接近"源版本绑定"的：键里有源，但没有定义版本与写入模型，且需要调用方触发。已完成的数据项在重跑时被跳过，不看 LLM 配置，所以换模型不会失效任何东西。导出有 pydantic / COGX / json / graphml / cypher，且能导入 Mem0、Zep、Letta、LangMem——但没有 .mem。

- D = 部分（code）：Typed DataPoint nodes and triples with dataset/data/chunk source refs and chunk content hashes; span-level provenance ledger exists but is opt-in, off by default, and fills chunk_index instead of offsets.
- B = 0（code + documented；实际键分量：source (dataset_id, data_id, chunk_id, content hash), caller-triggered; no definition/prompt or model key）：Formally 0 but the strongest source binding among external engines: deleting a source cascades with ref-counting and caller-invoked update() re-derives only changed chunks; completed items are skipped on re-run whatever the LLM config, so model/prompt changes invalidate nothing.
- T1 = 不可判定（code）：Global LLM setting, not recorded; re-derivation needs manual forget(memory_only=True) + re-cognify (raw data retained, so runnable); not in E5.
- T2 = 不可判定（code + documented）：Most portable engine audited: COGX archive (facts with valid_at/invalid_at/provenance, raw nodes verbatim) plus importers from Mem0, Zep/Graphiti, Letta, LangMem; but no .mem path and no carryover measurement.
- T3 = 不可判定（documented）：BEAM scores without full-context/oracle baseline.
- T4 = 不可判定（documented）：Not reported.
- 证据：[CG-CODE] https://github.com/topoteretes/cognee @c4cd8ceb9509dff6bddfabdadbeab7cc040bc32b: cognee/infrastructure/engine/models/DataPoint.py; cognee/infrastructure/databases/provenance/source_refs.py; cognee/infrastructure/databases/unified/provenance_delete_planner.py; cognee/api/v1/update/update.py, incremental.py; cognee/api/v1/forget/forget.py; cognee/modules/pipelines/operations/run_tasks_data_item.py ~L197-207; cognee/modules/pipelines/models/PipelineRun.py; cognee/modules/cognify/config.py L22; cognee/api/v1/export/export.py; cognee/modules/migration/cogx.py; cognee/modules/migration/sources/{mem0,zep,letta,langmem,cogx_archive}.py；[CG-DOCS] https://docs.cognee.ai/core-concepts/main-operations/forget; https://docs.cognee.ai/python-api/update; https://docs.cognee.ai/core-concepts/further-concepts/cogx.md; https://docs.cognee.ai/examples/migrate-memory-systems.md；[CG-BEAM] cognee/eval_framework/beam/REPORT.md §4.1 @c4cd8ceb9509dff6bddfabdadbeab7cc040bc32b；[GH-SEARCH] GitHub code search 'portable-memory repo:<r>' for mem0ai/mem0, getzep/graphiti, letta-ai/letta, topoteretes/cognee, plastic-labs/honcho, supermemoryai/supermemory, khoj-ai/khoj, MemTensor/MemOS, vectorize-io/hindsight; plus grep of the cloned repos；[E25] reference/research_2/evidence_registry.jsonl#E25-E26; reference/research_1/evidence/sources.json#S-COGNEE

### Honcho

*Honcho (Plastic Labs; managed or self-hosted)* — 拥有者：agent（developer platform keyed by workspace/peer; vendor on managed service; person when self-hosted）

派生"结论"是自然语言文本加类型化 level，内部存消息 id 区间（按批而非按结论），API 不返回。去重按嵌入距离与词重叠启发式替换；dreamer LLM 可删除过时结论；消息不可修订。无导出端点。

- D = 部分（code）：Conclusions are free text with a typed level; they carry an internal batch-level message-id range (not exposed by the API); no spans or parsed values.
- B = 0（code + documented；实际键分量：internal message-id range (batch-level); no key）：No key: heuristic dedup replacement and an LLM 'dreamer' deletes/replaces outdated conclusions; messages are immutable (sources cannot be revised); session deletion cascades only to explicit conclusions; model not stored.
- T1 = 不可判定（code + inferred）：Models set in operator config, not recorded, no re-derivation; messages retained so a full re-derive by re-ingest is possible; not in E5.
- T2 = 不可判定（code）：No export endpoint or format; list APIs return conclusions (without message ids) and messages; migrations re-ingest via the API; no .mem.
- T3 = 不可判定（documented）：Vendor reports accuracy above full context on LongMem-S/LoCoMo; different metric and world, not verified.
- T4 = 不可判定（documented）：Vendor reports ~11% of full-context tokens, which would clear c = 0.5 if it held here; not measured.
- 证据：[HO-CODE] https://github.com/plastic-labs/honcho @2eb27b6cc0595d3f8deb693f0560e7241c2aeaff: src/models.py (Document, DocumentSource); src/schemas/internal.py (DocumentMetadata.message_ids); src/deriver/deriver.py; src/schemas/api.py (Conclusion); src/crud/document.py (_semantic_dup_decision); src/utils/agent_tools.py (delete_observations); src/routers/messages.py (update_message: metadata only); config.toml.example；[HO-DOCS] docs/v3/documentation/features/advanced/{dreaming,deleting-data,evidence}.mdx; docs/v3/guides/migrations/mem0.mdx @2eb27b6cc0595d3f8deb693f0560e7241c2aeaff；[HO-BENCH] https://plasticlabs.ai/blog/research/Benchmarking-Honcho；[GH-SEARCH] GitHub code search 'portable-memory repo:<r>' for mem0ai/mem0, getzep/graphiti, letta-ai/letta, topoteretes/cognee, plastic-labs/honcho, supermemoryai/supermemory, khoj-ai/khoj, MemTensor/MemOS, vectorize-io/hindsight; plus grep of the cloned repos

### Supermemory

*Supermemory (hosted app/API; self-hosted local binary)* — 拥有者：厂商（agent for developer API (containerTag); person only for the self-hosted binary）

记忆有版本链（isLatest、parentMemoryId）与 updates / extends / derives 关系，新矛盾事实替代旧事实但旧版保留，另有 forgetAfter 到期——归入"仅时间有效性"（版本链变体，无事件时间区间）。证据来自客户端校验 schema 与文档，服务端引擎不公开。文档更新触发全量重处理，但对派生记忆的影响未写明。连接器只在云端版本。

- D = 部分（inferred）：Free-text memories with typed relations (updates/extends/derives) and document-level source links without span or revision (client validation schema; server engine not public).
- B = 仅时间有效性（inferred；实际键分量：version chain (isLatest, parentMemoryId) + forgetAfter; document id without revision; no definition or model key）：A newer contradicting memory supersedes the old one through an 'updates' relation (old kept, isLatest=false); expiry via forgetAfter; document update triggers full reprocessing but the effect on derived memories is undocumented.
- T1 = 不可判定（documented）：Cloud extraction uses proprietary models; self-hosted uses the configured model; neither recorded per memory.
- T2 = 不可判定（documented）：No export format (list APIs incl. version history; 'you can always export your data' without a documented mechanism); Mem0/Zep migration scripts only; no .mem.
- T3 = 不可判定（documented）：Vendor LongMemEval figures vs full context use a recall metric; not comparable.
- T4 = 不可判定（documented）：Vendor reports ~720 context tokens; not measured.
- 证据：[SM-CODE] https://github.com/supermemoryai/supermemory @cfa6c7cb17476d19ea896867406c80e8186a72ec: packages/validation/schemas.ts ~L239-295；[SM-DOCS] apps/docs/{concepts/graph-memory,recall/memory-operations,ingestion/document-operations,connectors/overview,self-hosting/local-vs-enterprise,overview/comparison}.mdx @cfa6c7cb17476d19ea896867406c80e8186a72ec；[SM-BENCH] https://supermemory.ai/research; README 'Benchmarks'；[GH-SEARCH] GitHub code search 'portable-memory repo:<r>' for mem0ai/mem0, getzep/graphiti, letta-ai/letta, topoteretes/cognee, plastic-labs/honcho, supermemoryai/supermemory, khoj-ai/khoj, MemTensor/MemOS, vectorize-io/hindsight; plus grep of the cloned repos

### Khoj

*Khoj (self-hostable; app.khoj.dev)* — 拥有者：人（self-hosted is the primary pitch; vendor on app.khoj.dev）

两个存储：文件 RAG 索引（按文件哈希增量重嵌入）与 2.0.0-beta.25 起的长期记忆（每轮对话后 LLM 决定新增 / 删除事实）。记忆没有对话引用，不记模型；只能导出对话。最后一次提交 2026-08-01，维护节奏不明。

- D = 0（code）：UserMemory is free text + embedding with no conversation/source reference.
- B = 0（code；实际键分量：none）：After each chat turn an LLM creates new facts and deletes related ones (update = delete + create); deleting a conversation cannot cascade (no link); no model field. The file index re-embeds changed files, but that is RAG, not derived memory.
- T1 = 不可判定（code）：Chat model user-switchable, not recorded; admin `manage_memories --force` can regenerate memories from conversation history, so a full re-derive is possible but unmeasured.
- T2 = 不可判定（code）：Memories readable via GET /api/memories (id, raw, created_at); only chats have an export (own JSON); no import; no .mem.
- T3 = 不可判定（inferred）：Not reported for memory.
- T4 = 不可判定（inferred）：Not reported.
- 证据：[KH-CODE] https://github.com/khoj-ai/khoj @ae229ca894c0b80ad84664afcfdde523b5e87057: src/khoj/database/models/__init__.py (UserMemory ~L855, Entry); src/khoj/routers/helpers.py (extract_facts_from_query, ai_update_memories); src/khoj/routers/api_memories.py; src/khoj/database/management/commands/manage_memories.py; src/khoj/processor/content/text_to_entries.py; src/khoj/routers/api_chat.py (export_conversation)；[KH-REL] https://github.com/khoj-ai/khoj/releases；[GH-SEARCH] GitHub code search 'portable-memory repo:<r>' for mem0ai/mem0, getzep/graphiti, letta-ai/letta, topoteretes/cognee, plastic-labs/honcho, supermemoryai/supermemory, khoj-ai/khoj, MemTensor/MemOS, vectorize-io/hindsight; plus grep of the cloned repos

### Opal

*Opal (arXiv 2604.02522, private memory for personal AI)* — 拥有者：厂商（provider-hosted enclave + ORAM; the user holds only a key and a counter）

UC Berkeley / Google DeepMind，只有 v1。TEE + ORAM，由助手提供方托管，面向"数百万用户"部署，所以拥有者记为厂商（计划预期为人）。KG 只存元数据，按规则从应用元数据抽取；LLM 摘要每 5 次摄入生成一次。过期按访问刷新的 TTL，不处理源修订。

- D = 部分（paper）：KG nodes/links are deterministically extracted from application metadata (no LLM); LLM summaries are linked to the artifacts they compress; no spans on derived items.
- B = 0（paper；实际键分量：artifact links only; no revision, definition or model key）：Items expire by a TTL in logical operations refreshed on access (LRU-like); summaries outlive the chunks they compress; source edits, deletion requests and model changes are not handled.
- T1 = 不可判定（paper）：Two model stacks evaluated; the model is not recorded per item; no code.
- T2 = 不可判定（paper）：Export/portability not discussed; no code release.
- T3 = 不可判定（paper）：No full-context or oracle baseline (only vs semantic search and Graphiti).
- T4 = 不可判定（paper）：Cost compared with a secure in-memory baseline, not a full re-read.
- 证据：[OPAL] arXiv:2604.02522v1 §3.2.1, §3.3.1, §3.3.3, §6.2-6.3, Table 1；[BG-CARDS] docs/background/IDEA_PAIN_POINTS_AND_CARDS_ZH.md §1.1 I1, I2, I5, I8, I9; §1.2 L9, L12 @56876e3634ca9493d9511e67d98e935be81cb887

### Plain RAG

*Plain RAG (chunks + embedding index; our baseline)* — 拥有者：人（whoever holds the sources; our baseline in src/memshift/adapters/baselines.py）

我们的基线（分块 + Qwen3-Embedding-0.6B + 同一读者）。无派生状态，T1、T2 按构造通过（未测量）。计划预期 T4"不通过"，但按 prereg 的读者预算 4,096 token 对约 2.2M token 的全量重读，服务成本远低于 0.5×，此格很可能通过（推断，待 E3）。

- D = 不适用（code）：No derived items: only source chunks and their embeddings.
- B = 不适用（code；实际键分量：n/a）：No derived items; the index is rebuilt from the sources as of the probe time.
- T1 = 通过（by construction）：No writer model in the path; theta_e changes are a separate axis (E7).
- T2 = 通过（by construction）：Nothing derived to lose: .mem resource/chunk kinds carry source text and receivers re-embed from text; not exercised by E6 (no RAG arm).
- T3 = 待 E3（pending experiment）：Measured by E3 (system list includes Plain RAG).
- T4 = 待 E3（pending experiment）：Measured by E3; with a 4,096-token reader budget against a ~2.2M-token full re-read the inequality is likely non-binding (inferred).
- 证据：[CODE-BASELINES] src/memshift/adapters/baselines.py @56876e3634ca9493d9511e67d98e935be81cb887 (ZeroSystem L56-72; Oracle L74-89; PlainRAG L164-)；[PM-SPEC-EMB] Spec/portable-memory-spec.md §2 kinds table (resource, chunk) and 'Embeddings & engine-agnostic representation' @f341f0703d739edfdf94eba58e66a4cc7c925df7；[CODE-NOEXPORT-OURS] grep over src/memshift @56876e3634ca9493d9511e67d98e935be81cb887；[PREREG-V0] prereg/v0.md §1 and E3 rule @56876e3634ca9493d9511e67d98e935be81cb887；[PLAN-S3] docs/MEMSHIFT_RESEARCH_PLAN_ZH.md §2.2, §3.1-§3.4, §6 E2/E3/E5/E6 @56876e3634ca9493d9511e67d98e935be81cb887

### Solid pod

*Solid pod (Solid Protocol v0.11; W3C LWS 1.0 draft)* — 拥有者：人（stated design intent: data stays the user's across providers and apps）

只有源没有知识。W3C LWS 1.0 工作草案（2026-09-21）的可移植性一节仍留空；用例要求整库迁移，但尚无规范。T3 的"不通过"是推断：单独的 pod 没有给读者的查询路径，加上检索层就变成零系统或 RAG。

- D = 不适用（documented）：Stores resources that apps write; neither Solid nor LWS defines derived or inferred memory.
- B = 不适用（documented；实际键分量：n/a (ETags on resources, not on derived items)）：No derived items.
- T1 = 通过（by construction）：No writer model and no derived items.
- T2 = 通过（by construction）：No derived state to lose; resources are retrievable over HTTP. Caveat: no standardized bulk export/migration (LWS §15 Portability left blank), so leaving needs ad hoc tooling.
- T3 = 不通过（inferred）：A pod alone offers storage and access control but no query path for a reader; any serving layer on top turns it into the zero system or RAG.
- T4 = 不可判定（inferred）：No serving path, so serving cost is undefined.
- 证据：[SOLID-PROTO] https://solidproject.org/TR/protocol；[SOLID-ABOUT] https://solidproject.org/about；[LWS] https://www.w3.org/TR/lws10-core/; https://www.w3.org/TR/lws-ucs/; https://www.w3.org/2024/09/linked-web-storage-wg-charter.html；[E01] reference/research_2/evidence_registry.jsonl#E01；[PLAN-S3] docs/MEMSHIFT_RESEARCH_PLAN_ZH.md §2.2, §3.1-§3.4, §6 E2/E3/E5/E6 @56876e3634ca9493d9511e67d98e935be81cb887

### User as Code

*User as Code (arXiv 2606.16707)* — 拥有者：agent（'a directory in the agent's workspace'; person if the user runs it）

两阶段：LLM 先把每次会话抽成只追加的平铺事实（带 [session, date]），再由 LLM 从事实列表整体重写类型化 Python 视图（仓库版本只看前 30,000 字符）。类型化状态本身不带源引用，所以 D 记为部分而非计划预期的 1；任何新会话都让整份视图重建，没有逐项键。换模型只重写视图，旧模型写的事实不重派生，所以 T1 不能按构造判通过。

- D = 部分（code + paper）：Typed Python dataclass values, but the typed state carries no source reference (only the flat facts carry [session, date]); no deterministic check of values against sources.
- B = 0（code + paper；实际键分量：none (session id on facts only)）：Phase 1 appends flat facts, never overwritten; Phase 2 regenerates the whole typed view whenever any new session arrives (full rebuild, no per-item key, no definition or model version).
- T1 = 不可判定（code）：Model is an environment constant, not recorded; a new model rebuilds the view from facts written by the old model on the next ingest, so re-derivation is only partial.
- T2 = 不可判定（code）：Plain Python files in a directory (copyable) but no exporter/importer, no provenance fields, no .mem.
- T3 = 不可判定（paper）：LOCOMO 78.8% vs full context 79.8% (judged accuracy, within 1 pp) and 99% on aggregate queries: suggestive of clearing a 3 pp floor, but a different metric, baseline and reader.
- T4 = 不可判定（paper）：Per-query cost below FC+REPL (1.2-2.3 vs 1.9-37.8 m$): suggestive, not our cost definition.
- 证据：[UAC-PAPER] arXiv:2606.16707v1 §3.2, §5, Tables 1, 3, 5, 6；[UAC-CODE] https://github.com/19PINE-AI/user-as-code @3d473617a27e7ace8cbbde531b667d515864c21e: experiments/user_as_code_v5.py L29-31 (MODEL = KRILL_MODEL), L89 (_code_stale on ingest), L129 ([session, date] prefix), L166-218 (structure), L183-184 (30,000-char prefix)；[BG-CARDS] docs/background/IDEA_PAIN_POINTS_AND_CARDS_ZH.md §1.1 I1, I2, I5, I8, I9; §1.2 L9, L12 @56876e3634ca9493d9511e67d98e935be81cb887

### 零系统（ours）

*Zero system (ours: sources + lexical index + fixed-budget re-read)* — 拥有者：人

E1 开发集运行里已作为基线行出现，但 E3 尚未运行（STATUS 阶段 1 记录）。

- D = 不适用（code）：No derived store.
- B = 不适用（code；实际键分量：n/a）：No derived store; every probe re-reads the sources granted at probe time.
- T1 = 通过（by construction）：No writer model in the path; E2 does not include the zero system, so this is not measured.
- T2 = 待 E6（pending experiment）：Expected to pass by construction (sources carried verbatim as .mem episodes/resources), but no export path exists and E6 has no zero-system arm; add a cheap arm or state it by construction.
- T3 = 待 E3（pending experiment）：E3 criterion (a)/(b) with epsilon = 3 pp.
- T4 = 待 E3（pending experiment）：E3 criterion (c), c = 0.5; with a 4,096-token reader budget vs a ~2.2M-token full re-read it is likely non-binding (inferred).
- 证据：[CODE-BASELINES] src/memshift/adapters/baselines.py @56876e3634ca9493d9511e67d98e935be81cb887 (ZeroSystem L56-72; Oracle L74-89; PlainRAG L164-)；[PREREG-V0] prereg/v0.md §1 and E3 rule @56876e3634ca9493d9511e67d98e935be81cb887；[STATUS-S1] docs/STATUS.md 'Gate G0' and 'Stage 1 run log' @56876e3634ca9493d9511e67d98e935be81cb887；[CODE-E6] src/memshift/experiments/e6.py module docstring @56876e3634ca9493d9511e67d98e935be81cb887；[CODE-NOEXPORT-OURS] grep over src/memshift @56876e3634ca9493d9511e67d98e935be81cb887；[PLAN-S3] docs/MEMSHIFT_RESEARCH_PLAN_ZH.md §2.2, §3.1-§3.4, §6 E2/E3/E5/E6 @56876e3634ca9493d9511e67d98e935be81cb887

### HoldMem (1,1)

*HoldMem (D=1, B=1) (ours)* — 拥有者：人

E1 开发集已运行（检测 / 修复）；E2、E3、E6 尚无结果。E6 驱动已写好：导出带 / 不带派生记录，只导入到第二个 HoldMem 实现。.mem 0.3.0 下 fact 上的 ext 在导入时被丢弃（已观测），派生记录必须走自定义 kind 或 episode ext。

- D = 1（code）：Item carries (parse, spans); the checker verifies citations against live source revisions and the kappa evaluator; rule fallback; else unverifiable.
- B = 1（code；实际键分量：(scope epochs or cited-cluster epochs, kappa_version, theta_w)）：key_at compares the stored key with the current (scope epoch, kappa version, theta_w); any change -> invalid -> re-derive with the current writer.
- T1 = 待 E2（pending experiment）：E2 (S_wr) measures carryover under a writer switch with theta_w in the key.
- T2 = 待 E6（pending experiment）：E6 driver exists (export with/without derivation record, import into a second HoldMem); the derivation record must travel as a custom record kind or episode ext because fact ext is dropped on import.
- T3 = 待 E3（pending experiment）：E3 static ContractValid vs oracle.
- T4 = 待 E3（pending experiment）：E3 serving cost vs full re-read.
- 证据：[CODE-HOLDMEM] src/memshift/holdmem/system.py @56876e3634ca9493d9511e67d98e935be81cb887 (CellConfig L52-60; key_at L137-153; set_writer L257-259; maintain L261-314)；[CODE-CHECKER] src/memshift/holdmem/checker.py module docstring @56876e3634ca9493d9511e67d98e935be81cb887；[CODE-MEMEXPORT] src/memshift/adapters/mem_export.py module docstring; export_holdmem L244; import_to_holdmem L444; import_to_mem0 L529 @56876e3634ca9493d9511e67d98e935be81cb887；[CODE-EXTSURV] src/memshift/adapters/mem_export.py ext_survival L377-420; tests/test_native_offline.py::test_ext_survival_by_carrier L405-429 @56876e3634ca9493d9511e67d98e935be81cb887；[CODE-E6] src/memshift/experiments/e6.py module docstring @56876e3634ca9493d9511e67d98e935be81cb887；[STATUS-S1] docs/STATUS.md 'Gate G0' and 'Stage 1 run log' @56876e3634ca9493d9511e67d98e935be81cb887；[PREREG-V0] prereg/v0.md §1 and E3 rule @56876e3634ca9493d9511e67d98e935be81cb887

### HoldMem (0,1)

*HoldMem (D=0, B=1) (ours)* — 拥有者：人

- D = 0（code）：No parse/spans; the value exists only in s_k.
- B = 1（code；实际键分量：(scope epochs, kappa_version, theta_w)）：Same key and invalidation as (1,1); repair is re-derivation without a checker.
- T1 = 待 E2（pending experiment）：E2 measures per-item survival predicted by per-item agreement rate.
- T2 = 待 E6（pending experiment）：Predicted by §2.3 (extension preserves B); E6 currently exports only (1,1), so this cell needs an extra arm.
- T3 = 待 E3（pending experiment）：Identical to (1,1) before any intervention by construction (byte-identical read path, hash-asserted); E3 runs (1,1) only.
- T4 = 待 E3（pending experiment）：Same read path as (1,1).
- 证据：[CODE-HOLDMEM] src/memshift/holdmem/system.py @56876e3634ca9493d9511e67d98e935be81cb887 (CellConfig L52-60; key_at L137-153; set_writer L257-259; maintain L261-314)；[CODE-E6] src/memshift/experiments/e6.py module docstring @56876e3634ca9493d9511e67d98e935be81cb887；[STATUS-S1] docs/STATUS.md 'Gate G0' and 'Stage 1 run log' @56876e3634ca9493d9511e67d98e935be81cb887

### HoldMem (1,0)

*HoldMem (D=1, B=0) (ours)* — 拥有者：人

- D = 1（code）：Checker on write; B=0 means maintenance appends new checked items.
- B = 0（code；实际键分量：none (append-only)）：Append-only: new items derived from the delta; old items never invalidate.
- T1 = 待 E2（pending experiment）：T1's precondition (theta_w in the key) is unmet for B=0; E2 reports survival, predicted ~1 because nothing is re-derived, which is not equivalence under re-derivation.
- T2 = 待 E6（pending experiment）：E6 currently exports only (1,1).
- T3 = 待 E3（pending experiment）：Identical to (1,1) before intervention by construction.
- T4 = 待 E3（pending experiment）：Same read path as (1,1).
- 证据：[CODE-HOLDMEM] src/memshift/holdmem/system.py @56876e3634ca9493d9511e67d98e935be81cb887 (CellConfig L52-60; key_at L137-153; set_writer L257-259; maintain L261-314)；[CODE-E6] src/memshift/experiments/e6.py module docstring @56876e3634ca9493d9511e67d98e935be81cb887；[STATUS-S1] docs/STATUS.md 'Gate G0' and 'Stage 1 run log' @56876e3634ca9493d9511e67d98e935be81cb887

### HoldMem (0,0)

*HoldMem (D=0, B=0) (ours)* — 拥有者：人

- D = 0（code）：No parse/spans.
- B = 0（code；实际键分量：none (append-only)）：Append-only; the memory-layer-like cell.
- T1 = 待 E2（pending experiment）：As (1,0): T1's precondition is unmet for B=0; E2 reports survival only.
- T2 = 待 E6（pending experiment）：E6 currently exports only (1,1).
- T3 = 待 E3（pending experiment）：Identical to (1,1) before intervention by construction.
- T4 = 待 E3（pending experiment）：Same read path as (1,1).
- 证据：[CODE-HOLDMEM] src/memshift/holdmem/system.py @56876e3634ca9493d9511e67d98e935be81cb887 (CellConfig L52-60; key_at L137-153; set_writer L257-259; maintain L261-314)；[CODE-E6] src/memshift/experiments/e6.py module docstring @56876e3634ca9493d9511e67d98e935be81cb887；[STATUS-S1] docs/STATUS.md 'Gate G0' and 'Stage 1 run log' @56876e3634ca9493d9511e67d98e935be81cb887

### Hindsight

*Hindsight (vectorize.io memory banks)* — 拥有者：agent（developer memory bank）

起点审计判为"符合（memory bank 范围）"。源文档替换 / 删除会级联重整合，观察带逐字引文——是除 HoldMem 外最接近 D=1、源级 B 的系统之一，建议补进 E5 候选。

- D = 部分（documented）：Observations track supporting evidence with exact quotes and a proof count; no typed parse or deterministic checker documented.
- B = 0（documented；实际键分量：source document id (replace/delete cascade + re-consolidation); no definition or model key）：Formally 0, but re-retaining a document_id replaces the document and all its memories and deleting it deletes derived observations; writer model per item undocumented.
- T1 = 不可判定（documented）：Not documented; not measured.
- T2 = 不可判定（documented）：Complete own-format ZIP (export-bank/import-bank) between Hindsight instances, i.e. the same implementation; no .mem.
- T3 = 不可判定（inferred）：Not measured.
- T4 = 不可判定（inferred）：Not measured.
- 证据：[HS-DOCS] https://github.com/vectorize-io/hindsight @8924a5bcfd6ff64fb20cace098021a3b61e76391: hindsight-docs/docs/developer/api/retain.mdx L116-133; observations.mdx L14, L257-269; api/documents.mdx L122-125; admin-cli.md L395-460；[E21] reference/research_2/evidence_registry.jsonl#E21-E22；[RA2-AUDIT] reference/research_2/system_audit.json

### OpenViking

*OpenViking (resource watches, context compilation)* — 拥有者：agent（person if self-hosted）

- D = 部分（documented）：Summary layers keyed to directory URIs with generated_by metadata; no spans.
- B = 0（documented；实际键分量：source directory via watches (periodic re-ingest); no model/definition key）：Formally 0: parents refresh when a child's text changes (directory granularity, batched at 10%); no model id.
- T1 = 不可判定（documented）：Not documented.
- T2 = 不可判定（documented）：.ovpack own export format; no .mem.
- T3 = 不可判定（inferred）：Not measured.
- T4 = 不可判定（inferred）：Not measured.
- 证据：[OV-DOCS] https://github.com/volcengine/OpenViking @1f4f7039fc394c5d04637828166f4e4e74e249e0: docs/en/api/15-watches.md; docs/en/concepts/03-context-layers.md L78-153; docs/en/api/14-ovpack.md；[E23] reference/research_2/evidence_registry.jsonl#E23-E24；[RA2-AUDIT] reference/research_2/system_audit.json

### MemOS

*MemOS (self-hosted modes)* — 拥有者：agent

- D = 部分（code）：Textual memories carry SourceMessage refs (message id, snippet; spans optional).
- B = 0（code；实际键分量：version/history chain; no key）：LLM-decided updates with versioned history; no writer-model field found.
- T1 = 不可判定（code）：Not recorded; not measured.
- T2 = 不可判定（code）：MemCube dump/load own format; no .mem.
- T3 = 不可判定（inferred）：Not measured.
- T4 = 不可判定（inferred）：Not measured.
- 证据：[MOS-CODE] https://github.com/MemTensor/MemOS @a7367d07e55db61099f7b4e2c1108bc5831a24f3: src/memos/memories/textual/item.py L16-45 (SourceMessage), L49-127 (status, version, history); src/memos/mem_cube/general.py L92-126 (dump/load)；[E28] reference/research_2/evidence_registry.jsonl#E28; reference/research_1/evidence/sources.json#S-MEMOS；[RA2-AUDIT] reference/research_2/system_audit.json

### GraphRAG

*GraphRAG (incremental update path)* — 拥有者：agent（developer indexing pipeline, not personal memory）

起点审计的 f40e9a26 结论在 769542f 仍成立（标题驱动增量）。

- D = 部分（code）：Entities/relationships carry text_unit_ids -> document ids; no spans.
- B = 0（code；实际键分量：document title (for incremental diff); no revision key）：Incremental update diffs by title and consumes only new inputs; a document revised under the same title is neither re-derived nor removed.
- T1 = 不可判定（inferred）：Not measured.
- T2 = 不可判定（code）：Parquet outputs, no importer.
- T3 = 不可判定（inferred）：Not measured.
- T4 = 不可判定（inferred）：Not measured.
- 证据：[GRAG-CODE] https://github.com/microsoft/graphrag @769542fbf1d8e5b4c6a8677fefc34621c87894c5: packages/graphrag/graphrag/index/update/incremental_index.py L29-57; load_update_documents.py L62; docs/index/outputs.md；[E30] reference/research_2/evidence_registry.jsonl#E30；[RA2-AUDIT] reference/research_2/system_audit.json

### MemTX

*MemTX (arXiv 2607.23929)* — 拥有者：agent（research prototype）

- D = 部分（paper）：Records carry evidence, provenance and validity intervals; no span-level checker documented.
- B = 0（paper；实际键分量：derived-from dependency graph with cascade on retraction; no model/definition key）：Formally 0: typed cascading repair on retraction, but invalidation on source revision or writer-model change is not documented.
- T1 = 不可判定（paper）：Not documented.
- T2 = 不可判定（paper）：Not documented.
- T3 = 不可判定（paper）：Not comparable.
- T4 = 不可判定（paper）：Not comparable.
- 证据：[MEMTX] arXiv:2607.23929v2 abstract, §3；[E12] reference/research_2/evidence_registry.jsonl#E12；[RA2-AUDIT] reference/research_2/system_audit.json

### Document-to-Database

*Document-to-Database / DataMosaic (PVLDB 19)* — 拥有者：agent（document-extraction prototype, not personal memory）

- D = 部分（paper）：Typed relational cells with a snippet-level provenance index; constraint checks but no documented value-vs-span check.
- B = 0（paper；实际键分量：source snippets (two-way provenance) + constraint re-evaluation; no writer-model key）：Formally 0 but closest external system to B=1: document deltas re-extract affected tuples, tuples losing all evidence are retracted, constraints are re-evaluated; extractor model not recorded.
- T1 = 不可判定（paper）：Not studied.
- T2 = 不可判定（paper）：Not studied.
- T3 = 不可判定（paper）：Not comparable.
- T4 = 不可判定（paper）：Not comparable.
- 证据：[D2DB] PVLDB 19(9):2522-2535 (2026), doi:10.14778/3819518.3819568, §3, evaluation Table 8；[E29] reference/research_2/evidence_registry.jsonl#E29；[RA2-AUDIT] reference/research_2/system_audit.json

### AnnoIndex

*AnnoIndex (arXiv 2608.13384)* — 拥有者：agent（research prototype）

- D = 部分（paper）：Document-level typed attribute values; no spans.
- B = 0（paper + inferred；实际键分量：none described）：Re-annotation on document or schema change is not addressed.
- T1 = 不可判定（paper）：Not studied.
- T2 = 不可判定（paper）：Not studied.
- T3 = 不可判定（paper）：Not comparable.
- T4 = 不可判定（paper）：Not comparable.
- 证据：[ANNO] arXiv:2608.13384v1；[E04] reference/research_2/evidence_registry.jsonl#E04；[RA2-AUDIT] reference/research_2/system_audit.json

### Agentic Context Cracking

*Agentic Context Cracking (arXiv 2608.31082)* — 拥有者：agent（research prototype）

- D = 部分（paper）：Cracked objects carry evidence <document, supporting region>; region form unspecified.
- B = 0（paper；实际键分量：none (static corpus)）：Static corpus assumed; structures are reused across model upgrades, i.e. no model keying.
- T1 = 不可判定（paper）：Not studied.
- T2 = 不可判定（paper）：Not studied.
- T3 = 不可判定（paper）：Not comparable.
- T4 = 不可判定（paper）：Not comparable.
- 证据：[ACC] arXiv:2608.31082v2 §3, §5, §6；[E05] reference/research_2/evidence_registry.jsonl#E05；[RA2-AUDIT] reference/research_2/system_audit.json

### DuckDB / pgvector / LOTUS

*DuckDB / pgvector / LOTUS alone (components)* — 拥有者：不适用（组件）

- D = 不适用（documented）：Components without derived personal memory.
- B = 不适用（documented；实际键分量：n/a）：No derived items.
- T1 = 不可判定（documented）：Not a data layer by itself.
- T2 = 不可判定（documented）：Not a data layer by itself.
- T3 = 不可判定（documented）：Not a data layer by itself.
- T4 = 不可判定（documented）：Not a data layer by itself.
- 证据：[COMPONENTS] https://duckdb.org/; https://github.com/pgvector/pgvector; https://github.com/lotus-data/lotus；[E01] reference/research_2/evidence_registry.jsonl#E01；[RA2-AUDIT] reference/research_2/system_audit.json

## 证据编号

全部证据见 `evidence.jsonl`。kind：doc = 文档或网页；code = 公开仓库在固定提交下的路径，或本仓库；paper = arXiv 或会议论文；record = `reference/` 或 `docs/` 中已有的证据记录。OpenAI 的页面直接抓取返回 403，是经 r.jina.ai 阅读代理读取的，已在对应条目的 note 中注明。
