# runs

每次运行一个目录：`env.json`（镜像 id、模型 id、revision）、`manifest.json`（世界 id、种子、参数、代码 commit、SHA256）、结果 JSONL、`analysis/`。先 1 个世界跑通再放量。

JSONL 原始结果不进 git（留在计算主机，可由种子与代码提交重跑）；`stage*_*/` 是合并分析，其 manifest 列出来源运行与各自 manifest 的 SHA256。

Cognee 的 18 个原生运行（`e5_cognee_*`、`e5_cognee_noise_*`）的 `manifest.json` 每个 12–13 MB（内含摄入日志），不进 git，与原始结果一起留在计算主机；各自的 SHA256 记在 `stage4_n_e5` 的 manifest 中。

## 作废的运行（保留记录，不进任何合并分析）

| 运行 | 原因 | 取代者 |
|---|---|---|
| `e5_{hipporag,lightrag,memos}_*_20260929T2049…2052*`、`e5_{hipporag,lightrag,memos}_*_20260929T211*`（批 8）、`m1_google_gemma-4-12b-it_warm_20260929T2055*`、`m2_…_20260929T2055*`、`m3_…_20260929T2055*`、`l_render_warm_20260929T2055*` | 服务安排调整时停止（decisions.md 2026-09-29 两条：批量摄入、GPU 分配）；批 8 的运行在 KV 抢占与超时下几乎无进展；gemma 作业暂停，已完成调用在缓存中 | 21:5x UTC 之后的同名运行；gemma 作业在区块 N 之后重启 |
| `e5_*_20260929T202*`（20:21–20:24 UTC 启动的 90 个原生运行）、`m1_google_gemma-4-12b-it_warm_20260929T2016*`、`m2_google_gemma-4-12b-it_s*_20260929T2017*`、`m3_google_gemma-4-12b-it_s*_20260929T2017*`、`l_render_warm_20260929T2017*` | 服务引擎批量上限从 48 提到 128 时停止（退出码 137，decisions.md 2026-09-29 条）；已完成调用命中缓存 | 20:49 UTC 之后的同名运行 |
| `e5_{amem,memos,lightrag,hipporag,cognee}_s1000…s1014_20260929T2003…2005*`、`e5_*_noise_s1000-1002_20260929T200*`、`e5_whitebox_cited_short_s*_20260929T2006*`、`e5_baselines_short_do_s*_20260929T2006*`、`e2_rep_20260929T201144Z` | 区块 N 首批与批处理同时占用 GPU1，队列约 2,000，LightRAG 请求超时导致抽取失败（部署负载所致）；全部停止（退出码 137），按 decisions.md 2026-09-29 条重跑：原生系统单独占用 GPU1、LightRAG 超时放宽；批处理在原生系统之后重跑（已完成调用在缓存中） | 同名运行（20:3x UTC 之后） |
| `e1_synth_s1001-1005_20260929T150907Z`、`e1_synth_s1006-1010_20260929T150909Z`、`e1_synth_s1011-1015_20260929T150911Z`、`e1_synth_s1001-1003_*`、`e1_synth_s1004-1006_*`、`e1_synth_s1007-1009_*`、`e1_synth_s1010-1012_*`、`e1_synth_s1013-1015_*`（18:00 UTC 起） | 区块 C 放量：每容器 3–5 个世界并行 × 9 格 × 每格 48 个线程超过运行容器的进程上限（`can't start new thread`），退出码 1 或被手动停止以免同样崩溃；已完成的调用在请求缓存中 | `e1_synth_s{1001-1002,…,1015}_*`（每容器 ≤ 2 个世界，18:04 UTC 起） |
| `e1_synth_s1000_20260929T134908Z` | 区块 C 冒烟第一次：沙箱把重复导入已提供的 `datetime, timedelta` 判为不允许的语法，51 个程序请求中 49 个因此不可执行（D1s 可执行率 0%）；按 `prereg/decisions.md` 2026-09-29 条修正后重跑 | `e1_synth_s1000b_20260929T144647Z` |
| `e1_dev_a_*`、`e1_dev_b_*`、`e1_dev_cited_a_*` | 早于 D=1 兜底 s_k 注明定义版本的修正 | `e1_dev_v3_*` |
| `e1_dev_final_*`、`e1_dev_orphan_s0-4_*` | 早于 B=0 空增量不调用写入模型的修正 | `e1_dev_v3_*`、`e1_dev_orphan_v3_*` |
| `e4_dev_s0-1_*` | 事件抽样未分层（60 个探针里 57 个是 important_contact） | `e4_dev_v2_*` |
| `e5_measure_20260928T202249Z` | 旧驱动只在全部系统跑完后才写结果；Mem0 前缀在 E1 满载下约 80 秒 / 事件，Graphiti 排在其后要一天以上，停止（2026-09-28 23:26 UTC，Mem0 已摄入 138 个事件，未写出 t_i） | `e5_measure2_*` |
| `e5_measure_mem0_20260928T232633Z`、`e5_measure_graphiti_20260928T232633Z` | 启动时未设长上下文端点（`MEMSHIFT_NATIVE_LLM`），原生系统的调用落到 16k 服务上，Mem0 在第 33 个事件因超上下文退出；Graphiti 随之停止 | `e5_measure2_*` |
| `e7_theta_r_s1001-1004_20260929T075317Z` | 四个世界并行、写入与读者各一个客户端，线程数超过运行容器的进程上限（`can't start new thread`），退出码 1，未写 manifest；已完成的读者调用在请求缓存里，由重跑复用 | `e7_theta_r_s1001-1002_*`、`e7_theta_r_s1003-1004_*` |
| `e7_theta_r_s1000_20260929T075157Z`、`e7_theta_r_s1001-1002_20260929T082701Z`、`e7_theta_r_s1003-1004_20260929T082703Z` | 读者由 vLLM 0.20.0 镜像服务，与其余服务的引擎版本不同；按 `prereg/decisions.md` 2026-09-29 条在主镜像上重跑。只作引擎对照的"旧"一侧（`stage3_theta_r_engine/`） | `e7_theta_r030_*` |
| `e5_lightrag_s1000-1014_20260929T2202*Z`、`e5_lightrag_noise_s1000-1002_20260929T2202*Z`、`e5_amem_s1000-1014_20260929T2049*Z`、`e5_amem_noise_s1000-1002_20260929T2049*Z`（36 个，只在计算主机上） | 2026-09-30 03:26–03:28 UTC 在前缀摄入中停止，改为分段运行（`prereg/decisions.md` 2026-09-30 条）；其前缀存储被续跑运行复制后继续摄入，本身不写结果 | `e5_{lightrag,amem}_*_pre_*`（前缀）与 `e5_{lightrag,amem}_*_{src,def,orphan}_*`（分支） |
| `e5_amem_s1000-1014_pre_20260930T0327*Z`、`e5_amem_noise_s1000-1002_pre_20260930T0328*Z`（18 个，只在计算主机上） | 2026-09-30 03:53 UTC 为让 LightRAG 优先而停止（`prereg/decisions.md` 2026-09-30 条）；其前缀存储由之后的同名续跑运行复制后继续摄入 | 之后的 `e5_amem_*_pre_*` |
| `m1_google_gemma-4-12b-it_warm_20260930T030116Z`（只在计算主机上） | 2026-09-30 04:17 UTC 退出码 1：GPU0 的 gemma 服务队列约 1,000 个请求（约 126 个完成 / 分钟），写入请求在 600 秒客户端超时内 5 次都没排到，`LLM request failed after retries: timed out`；未写 manifest，已完成的调用在请求缓存中 | 同名 `m1_google_gemma-4-12b-it_warm_*` 重跑（GPU0 队列降到 300 以下或其余 gemma 作业剩 2 个时启动） |
| `e5_amem_s1013_src_20260930T0615*Z`、`e5_amem_s1000_src_20260930T1514*Z`（只在计算主机上） | 2026-09-30 16:05 UTC 退出码 1：GPU1 重启期间 A-Mem 负载控制器解冻了被冻结的容器，读者请求落在启动中的服务上（`Server disconnected without sending a response`，4 次重试后失败）；部署故障 | 同名 `e5_amem_s1013_src_*`、`e5_amem_s1000_src_*`，16:16 UTC 以同一参数从同一前缀快照重跑 |
| `e5_amem_s1013_src_20260930T1616*Z`、`e5_amem_s1000_src_20260930T1616*Z`（只在计算主机上） | 2026-09-30 17:32 UTC 为改用 GPU0 副本（8001）而终止（此前在 8003 的长队列后几乎停顿） | `e5_amem_s1013_src_20260930T1732*Z`、`e5_amem_s1000_src_20260930T1732*Z`（退出码 0） |
| `e5_whitebox_cited_short_s1000-1002_20260930T1607*Z`、`…_s1009-1011_20260930T1607*Z`、`…_s1012-1014_20260930T1607*Z`、`e5_baselines_short_do_s1000-1007_20260930T1607*Z`、`e5_baselines_short_do_s1008-1014_20260930T1607*Z`（只在计算主机上） | 退出码 1：批处理一次启动后 8003 队列超载，客户端请求超时（`LLM request failed after retries: timed out`） | 同名的 `…_20260930T1717*Z` / `…_20260930T1732*Z` 重跑（退出码 0） |
| `m1_read_google_gemma-4-12b-it_20260930T1607*Z`、`m1_read_meta-llama_llama-3_1-8b-instruct_20260930T1607*Z`、`m1_read_openai_gpt-oss-20b_20260930T1607*Z`、`m1_read_mistralai_ministral-3-14b-instruct-2512-bf16_20260930T1642*Z`、`e2_rep_20260930T160723Z`（只在计算主机上） | 退出码 1：被批处理准入控制器长时间冻结后解冻，HTTP 连接池里的连接已被服务端关闭（`Connection reset by peer`），或在 8001 从转发切换到副本的空窗内失败 | 批处理执行器以新进程重跑（`…_r2`；R 的 E2 为 `e2_rep_20260930T174558Z`） |
| `l_e1_s1000-1001_20260930T1607*Z` 至 `l_e1_s1008-1009_20260930T1607*Z`（5 个，只在计算主机上） | 被冻结约 1 小时后于 17:17 UTC 终止（退出码 137），以免解冻后因失效连接崩溃 | 批处理执行器的 `l_e1_*_r2` 重跑 |
| `e1_think_s1000-1001_20260930T192055Z`（只在计算主机上，无 manifest） | 思考写入者在 600 秒客户端超时下退出（退出码 1，decisions.md 2026-10-01 条）；已完成调用在缓存中 | 提交 `293ab60`（超时 1,800 秒）的 `e1_think_s1000-1001_20260930T201441Z` |
