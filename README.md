# 成人血压指南问答：自适应检索与证据核证

本项目实现面向医务人员的一般成人血压测量、筛查与诊断指南问答。系统从经整理的指南和系统综述条款中检索证据，回答时逐条验证引文标识与原文锚点，并在依据不足或问题超出服务范围时拒答。它不提供个人诊断、处方或治疗建议。

项目报告为 [`实验报告.md`](实验报告.md)。代码、语料、评测日志和图表均按本地归档保存；本 README 说明运行入口、两种推理引擎和结果复现方式。

## 环境与模型服务

项目使用 Python 3.11。命令行和评测依赖 NumPy 与 Requests；图表重算还需要 Matplotlib。

```bash
conda activate llm
pip install -r requirements.txt
```

模型服务由现有环境提供，项目不会启动或安装新的模型服务。`model_backend.Backend` 读取以下环境变量；未设置时使用表中的本机默认值。

| 变量 | 默认值 | 用途 |
|---|---|---|
| `EMBED_URL` | `http://127.0.0.1:8765` | 向量服务的 `/health` 和 `/embed` 接口 |
| `CHAT_URL` | `http://127.0.0.1:18882/v1` | OpenAI 兼容的聊天接口根路径 |
| `CHAT_MODEL` | `qwen3.6-35b-a3b` | 聊天模型名称 |

如果服务地址不同，请在启动命令的 shell 中覆盖变量。例如：

```bash
export EMBED_URL=http://127.0.0.1:8765
export CHAT_URL=http://127.0.0.1:18882/v1
export CHAT_MODEL=qwen3.6-35b-a3b
```

PowerShell 对应写法：

```powershell
$env:EMBED_URL = "http://127.0.0.1:8765"
$env:CHAT_URL = "http://127.0.0.1:18882/v1"
$env:CHAT_MODEL = "qwen3.6-35b-a3b"
```

嵌入服务需返回与 `/health` 相符的模型修订、last-token pooling 与归一化信息；查询和文档向量必须为 4,096 维。服务不可用或配置不兼容时，后端会报错，不会静默切换模型。

## 命令行问答

从项目根目录运行。默认使用保留的基线引擎：

```bash
python cli.py --query "NICE 对成人家庭血压监测的记录要求是什么？" --mode adaptive
```

若要运行改进引擎，显式选择 `--engine improved`：

```bash
python cli.py --engine improved --query "NICE 心律不齐时是否应改用人工听诊？" --mode single
```

两种引擎使用相同的问答语料和响应 API。基线 `clinical.System` 是主评测的冻结实现；`improved_clinical.System` 是独立改进分支，增加了由语料机构名和主题词共同支持的范围识别、只采用问题原文片段的证据需求规划、预置证据保留、按缺项分开补检，以及优先保留首轮已有引文支持的证据。改进引擎的差异和评估范围见报告；基线结果不应与扩展实验混报。

可选的 `--supplied ID1,ID2` 只接受 `data/evidence.jsonl` 中的记录标识。模式含义如下：

| 模式 | 行为 |
|---|---|
| `reuse` | 只核对调用方提供的证据，不检索 |
| `single` | 对原问题检索一轮 |
| `multi` | 规划缺项并执行两轮检索 |
| `adaptive` | 根据问题与已有证据选择复用、检索一轮或补检 |
| `adaptive_nocrit` | 使用自适应路由和确定性引用检查，关闭语义核证以作消融比较 |

完整回答是默认行为。改进引擎可用 `--allow-partial` 显式允许系统返回有依据的部分回答，并在答案中标出未获支持、未作答的方面：

```bash
python cli.py --engine improved --allow-partial --query "成人血压测量需要什么；第一天如何处理？" --mode single
```

`--params params.json` 可向路由传入历史开发日志选出的参数；常用参数包括 `initial_threshold`、`supplement_threshold`、`multi_threshold`、`initial_k`、`supplement_k`、`max_evidence`、`lexical_weight` 和 `force_action`。改进引擎还接受布尔值 `allow_partial`。不带 `--allow-partial` 时，部分回答仍会按原策略拒答。

输出为 JSON，包含答案、引用记录、来源、检索轮数、第一轮与最终证据、路由轨迹及诊断字段。改进引擎额外返回逐项计划与检索诊断。默认不会写入请求日志；可选 `--audit-log audit.jsonl` 记录经常见直接身份信息脱敏的请求、来源、模式与耗时。不要在查询中输入患者姓名或其他身份信息。

## 数据

- `data/evidence.jsonl`：80 条成人血压指南和系统综述证据记录，保存来源机构、版本、人群、定位、正文和主题。
- `data/questions.jsonl`：62 道问题，分为 36 道开发题和 26 道测试题。
- `data/challenge_questions.jsonl`：扩展挑战题，与主题库和主测试集分开记录。
- `data/sources.json`：来源清单与版本元数据。

`results/extended_manual_review.jsonl` 保存对扩展测试集26道自适应回答、6道边界题和10道关键问题的单轮/无语义核证对照的逐条人工证据核读。该核读用于比较回答陈述与所引语料条目，不是临床专家评审，也不读取评分标签。

语料构建入口为 `data/build_clinical_corpus.py`。此脚本没有命令行选项；运行时会重建 `evidence.jsonl` 和 `questions.jsonl`。只有有意重建数据时才运行它，并在覆盖前核对版本和冻结哈希。挑战题文件不由该脚本生成。

向量索引由系统根据语料和嵌入服务签名自动构建或增量更新，保存在 `index/clinical-vectors.npz`。它是可再生缓存，不纳入归档。评测的逐请求聊天缓存也可再生，不纳入归档；支持论文结论的 JSONL、JSON、CSV 和 Markdown 结果保留在 `results/`。

## 评测复现

以下命令均从项目根目录运行。评测会调用模型服务；图表重算只读取已有评测数据，不需要生成模型。为避免覆盖已归档结果和报告图，示例把复现输出写入新的结果子目录。

基线引擎主评测：

```bash
python eval/evaluate.py --project . --split all \
  --modes reuse single multi adaptive adaptive_nocrit \
  --output results/reproduced_main.jsonl --workers 2
```

可用 `--split dev` 或 `--split test` 选择数据划分，`--ids` 限定问题标识；`--workers` 接受 1 到 3。默认评测缓存位于 `results/chat_cache.sqlite3`。`--disable-replay-cache` 可直接使用物理模型后端进行调用计量。

改进引擎扩展评测：

```bash
python eval/extended_evaluate.py --project . \
  --questions data/challenge_questions.jsonl --split all \
  --modes single multi adaptive adaptive_nocrit \
  --output results/reproduced_challenge.jsonl --workers 1
```

加入 `--allow-partial` 才会评估部分回答策略。该实验有独立的 `results/extended_chat_cache.sqlite3`；缓存键包含改进引擎版本和模块哈希，避免混用不同引擎版本的响应。

对已生成的主评测日志重算汇总与四幅主结果图：

```bash
python eval/summarize.py --input results/reproduced_main.jsonl \
  --outdir results/reproduced_summary
python eval/plot_results.py --summary results/reproduced_summary/summary.json \
  --outdir results/reproduced_figures
```

对既有检索研究汇总重画候选深度图：

```bash
python eval/plot_retrieval.py results/retrieval_study.json \
  results/reproduced_figures/retrieval_depth.png
```

扩展检索扫描使用查询和已编码语料，不调用生成模型：

```bash
python eval/extended_retrieval_sweep_v2.py \
  --questions data/challenge_questions.jsonl --split all \
  --output-prefix results/reproduced_extended_retrieval
```

`plot_results.py` 和 `plot_retrieval.py` 需要 Matplotlib；示例使用独立输出目录，不覆盖当前报告引用的 `figures/`。其他实验脚本位于 `eval/`，评测输入、输出和元数据位于 `results/`。

扩展测试集与挑战集对照图由冻结的汇总 CSV 重绘。为避免覆盖报告中的 `图6`，先将绘图脚本和数据复制到临时输出目录，再在副本目录执行：

```bash
mkdir -p results/replot_extended_pipeline
cp figures/plot_extended_pipeline_metrics.py figures/extended_pipeline_metrics.csv \
  results/replot_extended_pipeline/
python results/replot_extended_pipeline/plot_extended_pipeline_metrics.py
```

PowerShell 可用以下命令将脚本和CSV复制到独立目录后重绘：

```powershell
New-Item -ItemType Directory -Path results/replot_extended_pipeline -Force
Copy-Item figures/plot_extended_pipeline_metrics.py, figures/extended_pipeline_metrics.csv `
  -Destination results/replot_extended_pipeline
python results/replot_extended_pipeline/plot_extended_pipeline_metrics.py
```

图像将写入 `results/replot_extended_pipeline/图6_扩展测试与挑战集证据覆盖.png`。

图7–图11由 `eval/extended_plot_results.py` 根据扩展检索排序、补检轨迹和完整金标writer诊断结果重算：

```bash
python eval/extended_plot_results.py --data-dir results \
  --out-dir results/reproduced_extended_figures
```

所需源文件为 `extended_retrieval_{test,challenge}_{multi,adaptive}.json`、`extended_trace_{test,challenge}_{multi,adaptive}.json`，以及 `extended_oracle_summary_{test,challenge}/summary.json`。脚本写入独立输出目录，不覆盖报告引用的图7–图11。

## 归档内容与缓存

归档包含推理引擎、CLI、评测与制图脚本、测试、来源材料、题库、报告、图表以及支持报告结论的实验输出。以下缓存与本机文件不会提交：

- `index/clinical-vectors.npz`：可再生向量索引；
- `results/*chat_cache.sqlite3`：精确请求复用缓存；
- Python 字节码、临时日志和本地环境配置。

主评测与扩展实验的原始 JSONL、汇总数据和逐项人工证据核读保留在 `results/`，以便复核报告中的统计和定性结论。运行时索引、聊天请求缓存、字节码及临时日志由 `.gitignore` 排除。
