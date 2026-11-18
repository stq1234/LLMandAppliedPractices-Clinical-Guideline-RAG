# 7.3 扩展实验交接数据

## 协议与口径

本轮使用冻结版 `improved_clinical` 0.2.0，证据库80条；原题库test 26题，独立challenge 24题。默认参数在dev adaptive 36题确认后冻结：`initial_threshold=0.35`、`supplement_threshold=0.25`、`multi_threshold=0.50`、`initial_k=5`、`supplement_k=5`、`max_evidence=8`、`lexical_weight=0.15`、`allow_partial=false`。未根据test或challenge调整参数。

常规评测每条只将 `query` 和 `supplied` 传入系统；gold IDs、expected groups、题型标签仅用于生成后的离线评分。固定模式为single、multi、adaptive、adaptive_nocrit。每批均以并发2通过ReplayBackend运行，原始响应日志中保留逐题 `trace`、首轮/最终证据、引用、诊断与逐逻辑请求成本。全部test四模式104条、challenge四模式96条均为 `status=ok`；没有模型请求异常。挑战集评价与test独立存档。

评分中的supported keypoint要求冻结 `expected_groups` 逐组文字匹配且完整金标引用；它是保守的机械验收口径，不是医学专家判断。语义audit在引用ID、来源显示项剥离后抽取答案数值、机构名、方向词及各expected group匹配情况，并附上金标原文、authority、population与locator。词项出现不代表数值、方向或来源归属正确；“通过/错误”需结合人工核读解释。

## 常规端到端模式结果

“联合通过”定义为可答题给出所有expected keypoints且具备完整gold citations；不可答和域外题要求正确拒答。token为每题平均prompt+completion，模型成本同时报告逻辑调用和真实服务调用；缓存命中只减少物理请求/等待，不改变逻辑提示与生成结果。

|集合|模式|联合通过|正确拒答|最终完整gold引用|均值轮数|逻辑调用/题|物理调用/题|缓存复用/题|token/题|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|test|single|4/22|4/4|14/22 (0.636)|0.88|2.92|2.65|0.27|3,588|
|test|multi|4/22|4/4|14/22 (0.636)|1.77|3.85|1.31|2.54|5,292|
|test|adaptive|4/22|4/4|14/22 (0.636)|1.15|3.38|0.27|3.12|4,117|
|test|adaptive_nocrit|4/22|4/4|16/22 (0.727)|1.08|1.81|0.50|1.27|1,673|
|challenge|single|0/20|4/4|9/20 (0.450)|0.88|2.58|2.54|0.04|3,199|
|challenge|multi|0/20|4/4|13/20 (0.650)|1.75|3.71|1.08|2.62|5,704|
|challenge|adaptive|0/20|4/4|13/20 (0.650)|1.58|3.62|0.21|3.42|5,478|
|challenge|adaptive_nocrit|1/20|4/4|14/20 (0.700)|1.50|1.75|0.62|1.12|1,885|

每个test模式的26条日志均有效，22道可答题中都只有4条联合通过；四种模式在26题整体配对联合结果上全为平局。challenge四模式24条均有效；adaptive_nocrit相对其他模式在一条可答题多出一个文字支持通过，pairwise为1 gain/0 loss/23 ties（24配对）；完整adaptive相对NoCrit为0 gain/1 loss/23 ties。challenge的0/20～1/20文字关键点通过率是本评分规则的结果，需与逐题semantic audit和人工核读区分。

初检需求以冻结 `initial_need` 为参照。test共22题要求检索：single、多轮各执行21题，adaptive20题，NoCrit19题；新引擎在D05等个别可答题上出现零轮前置拒答，保留为scope误拒。challenge 19题要求检索：single与multi各18题、adaptive18题、NoCrit17题执行检索。全部四题非可答样本在各模式都正确拒答。

## 固定证据库的无生成Top-K排序

候选检索扫表在80条冻结证据库内比较词项、稠密、混合、以及只在问题明确提及机构时应用的来源过滤；系统不读取gold，gold仅用于离线Top-K覆盖。数值是“全部必需gold IDs均进入Top-K”的题数。

|集合|方法|Top-3|Top-5|Top-8|
|---|---|---:|---:|---:|
|test (n=22)|词项|17/22|17/22|18/22|
|test|稠密|18/22|20/22|20/22|
|test|混合|21/22|21/22|21/22|
|test|显式机构过滤|20/22|21/22|22/22|
|challenge (n=20可答)|词项|18/20|18/20|19/20|
|challenge|稠密|18/20|18/20|19/20|
|challenge|混合|18/20|19/20|19/20|
|challenge|显式机构过滤|18/20|19/20|20/20|

题目中明确点名来源的查询数为test 21/22、challenge 19/20。对这些题使用显式来源过滤后，候选池中位数由80条分别降至27条和20条。逐题配对金标召回变化：test Top-3为下降1/不变20/提高0，Top-5全部21题不变，Top-8提高1/不变20；challenge Top-3全部19题不变，Top-5和Top-8各提高1题，其余不变。来源过滤并非普遍提升：test低深度一个查询有金标召回损失。

## 子查询检索与实际补检轨迹

静态子查询可行性是基于真实多轮trace中的follow-up query，逐query独立重新排序后合并Top-K；只有具有实际补检查询的可答题纳入分母，不把缺trace视为零召回。来源过滤下，test forced-multi在Top-3/5/8完整并集覆盖为19/21、19/21、20/21；test adaptive为6/8、6/8、7/8。Challenge forced-multi为15/19、17/19、18/19，challenge adaptive为11/15、13/15、14/15。该静态模拟说明子查询候选排序具备部分找回空间，不等同于系统实际补回。

实际日志trace单独给出：test forced-multi首轮完整gold覆盖19/22，21题触发补检；adaptive首轮18/22，8题补检。两个路由的补检都没有新增任何必要gold ID、也没有救回一个联合通过回答。Challenge forced-multi与adaptive首轮各为14/20，实际补检分别覆盖19题与15题，各在2题中新增gold（两例属于review_evidence），最终完整覆盖16/20，平均gold召回从0.825升至0.875；仍有0题被补检救回支持答案。这区分了“能找回部分材料”和“答案因补检变正确”。

## All-gold oracle writer/critic分离

Oracle臂每题把冻结完整gold IDs作为supplied，采用同一冻结planner与writer提示。`allgold_critic`通过常规mode=reuse完整跑critic/validator（0轮检索）；`allgold_writer_only`直接调用同一 `_plan` 与 `_write_claims`，绕过router、retriever、critic、claim-anchor validation及semantic validation，引用ID直接从原始writer claims抽取并格式化供机械评分。两臂均用ReplayBackend与相同版本key；gold只用于oracle输入，没有把expected groups作为提示。

|集合|Oracle臂|机械关键点+完整引用通过|完整gold引用ID覆盖|逻辑调用/题|物理调用/题|prompt+completion tokens/题|
|---|---|---:|---:|---:|---:|---:|
|test|完整核证|3/22|15/22|3.45|1.82|2,144|
|test|仅writer|4/22|22/22|2.00|0.09|1,033|
|challenge|完整核证|0/20|11/20|3.00|1.55|2,109|
|challenge|仅writer|0/20|20/20|2.00|0.20|1,171|

配对ID完整匹配。test中writer-only相对critic增加1个机械关键点通过；challenge两个arm都没有关键点通过。writer-only的citation覆盖来自未验证raw claims，不能当作系统可用引用或事实正确；这个oracle对照显示即使完整gold文本在上下文，答案writer/表达关键点仍有明显瓶颈，常规路由/召回不是唯一解释。`

先前尝试的 `adaptive_nocrit + force_action=reuse` 仍会在估计的supplied gap触发fallback retrieval（test 4/22、challenge 3/20发生），因此该日志仅保留为route pilot；最终固定证据对比排除这些行，以严格0轮的 `mode=reuse` critic和直接writer-only臂对照。

## 图表、数据与复现产物

本轮本地新增PNG为 `作业7.3_临床指南问答中的检索决策/figures/图7_初始检索TopK覆盖.png`、`图8_来源过滤逐题变化.png`、`图9_子查询TopK可行覆盖.png`、`图10_实际补检增量与答案救回.png`、`图11_Oracle固定证据Writer核证诊断.png`。绘图脚本与源JSON保存在a6000作业目录：`eval/extended_plot_results.py`；候选Top-K数据 `results/extended_retrieval_{test,challenge}_{multi,adaptive}.json`；实际trace recovery `results/extended_trace_{test,challenge}_{multi,adaptive}.json`；端到端summary `results/extended_{test,challenge}_summary_final/summary.json`；oracle summary `results/extended_oracle_summary_{test,challenge}/summary.json`。图7–11由同一脚本生成，字体使用Microsoft YaHei，300 dpi。a6000 `llm` 环境没有matplotlib，PNG在本机Python 3.12/Matplotlib 3.9.2生成，绘图脚本已远端py_compile验证。

新增评测脚本：`eval/extended_evaluate.py`、`eval/extended_oracle_writer_only.py`、`eval/extended_retrieval_sweep_v2.py`、`eval/extended_trace_recovery.py`、`eval/extended_semantic_audit.py`、`eval/extended_summarize.py`、`eval/enrich_extended_logs.py`、`eval/merge_oracle_runs.py`、`eval/extended_plot_results.py`。原始四模式日志为 `results/extended_test_modes.jsonl` (104行) 与 `results/extended_challenge_modes.jsonl` (96行)；固定证据有效oracle行为 `results/extended_oracle_valid_{test,challenge}.jsonl`。另有逐题semantic audit JSON/CSV与family/pair summary，供结果分析复核。

## 独立答案核读摘要

独立核读记录共52行：test adaptive主样本26题、10题的single/NoCrit对照20行、challenge边界题6行（C05、C07、C13、C16、C23、C24）。核读者依据题面、生成答案和被引证据直接阅读，未查看gold、expected或机械grading标签；该工作是课程项目质量核读，不构成临床专家意见。文件为 `results/extended_manual_review.jsonl`。

核读者判定test主样本的scope门禁25/26正确；22道可答题中7道被拒，其中D05的NICE-POST-01证据支持“先测坐位/卧位、后站立复测”，模型却因缺少明显血压域词而零轮拒答。T14引用的数值本身能在综述找到，但将筛查统计用于回答诊室确诊问题，需保留来源任务错配。T13的NoCrit答案遗漏AHA/ACC来源侧；T03的NoCrit在已有依据下能完整回答但semantic模式拒答；T21的NoCrit正确依据“或”解读，semantic模式拒答。这些案例说明引用锚点/数值出现与问题所需的来源、命题结构和范围对应关系需要分开判断。

六条challenge边界核读中，C05回答了四类阈值；C07、C13、C16在证据可答时仍拒绝；C23、C24正确拒绝儿科范围与个体处方要求。逐行结果保存在上述JSONL中，最终写作应将其与expected_groups的自动验收分开归因。
