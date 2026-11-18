from __future__ import annotations

import json
from pathlib import Path


OUT = Path(__file__).resolve().parent

SOURCES = [
    {
        "source_id": "NICE_NG136",
        "authority": "National Institute for Health and Care Excellence (NICE)",
        "title": "Hypertension in adults: diagnosis and management",
        "version": "NG136; published 2019-08-28; last updated 2026-02-26; current source page checked 2026-11-18",
        "url": "https://www.nice.org.uk/guidance/ng136/chapter/recommendations",
        "update_url": "https://www.nice.org.uk/guidance/ng136/chapter/update-information",
        "current_check": "NICE search-indexed recommendation and update pages plus 2026 guideline PDF; search page reports a February 2026 recommendation addition. The update page also lists an October 2026 wording clarification to 1.4.26, outside this diagnostic corpus.",
        "population": "Adults aged 18 years and older, including adults with type 2 diabetes for measurement and diagnosis; pregnancy is addressed in a separate NICE guideline.",
    },
    {
        "source_id": "AHA_ACC_2025",
        "authority": "American Heart Association / American College of Cardiology and collaborating societies",
        "title": "2025 AHA/ACC/AANP/AAPA/ABC/ACCP/ACPM/AGS/AMA/ASPC/NMA/PCNA/SGIM Guideline for the Prevention, Detection, Evaluation and Management of High Blood Pressure in Adults",
        "version": "Circulation 2025;152:e114–e218; published 2025-09-16; online version includes corrections dated 2025-11-10 and 2026-05-26; checked 2026-11-18",
        "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001356",
        "correction_urls": [
            "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001396",
            "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001448",
        ],
        "current_check": "The official online guideline is the 2025 adult guideline and retires the 2017 guideline. Its linked corrections affect introductory/resistant-hypertension and emergency-drug passages, not the measurement/classification facts used here.",
        "population": "Adults; the guideline is not a source for pediatric or pregnancy-specific diagnostic care. Use only the stated adult classification and measurement sections.",
    },
    {
        "source_id": "ISH_GLOBAL_2026",
        "authority": "International Society of Hypertension (ISH)",
        "title": "2026 International Society of Hypertension Global Hypertension Practice Guidelines",
        "version": "Journal of Hypertension 2026;44(12):2157–2188; published online 2026-10-20; checked 2026-11-18",
        "url": "https://doi.org/10.1097/HJH.0000000000004451",
        "current_check": "New global practice guideline; online article metadata and full indexed sections/tables were read. Issue date is December 2026; corpus freeze precedes the issue month but follows online publication.",
        "population": "The adult classification and diagnostic sections only. Pediatric, pregnancy, and individual medication decisions are outside this evidence set.",
    },
    {
        "source_id": "USPSTF_SCREEN_2021",
        "authority": "U.S. Preventive Services Task Force (USPSTF)",
        "title": "Hypertension in Adults: Screening — Final Recommendation Statement",
        "version": "Final recommendation dated 2021-04-27; current USPSTF topic page checked 2026-11-18 and remains the published recommendation",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/hypertension-in-adults-screening/",
        "current_check": "The USPSTF topic search lists the 2021 recommendation as published; the current recommendations-in-progress listing does not list an adult hypertension-screening update.",
        "population": "Adults aged 18 years and older without known hypertension; this screening recommendation does not cover children/adolescents or known hypertension follow-up.",
    },
    {
        "source_id": "USPSTF_REVIEW_2021",
        "authority": "USPSTF-commissioned evidence review; Agency for Healthcare Research and Quality (AHRQ)",
        "title": "Screening for Hypertension in Adults: Updated Evidence Report and Systematic Review for the U.S. Preventive Services Task Force",
        "version": "JAMA 2021;325(16):1657–1669; published 2021-04-27; evidence surveillance through 2021-03-26",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/document/final-evidence-summary/hypertension-in-adults-screening",
        "bibliographic_url": "https://pubmed.ncbi.nlm.nih.gov/33904862/",
        "current_check": "Full evidence-summary article and PubMed record read. This is an evidence synthesis, not a clinical guideline or a source of universal diagnostic thresholds.",
        "population": "Review of screening/confirmation accuracy in adults; included accuracy studies enrolled untreated adults or stratified by treatment and were conducted in very-high-HDI countries.",
    },
    {
        "source_id": "JAHA_AOBP_REVIEW_2025",
        "authority": "Journal of the American Heart Association; independent systematic review/meta-analysis",
        "title": "Revisiting Unattended Versus Attended Automated Office Blood Pressure Measurements: A Systematic Review and Meta-Analysis",
        "version": "J Am Heart Assoc. 2025;14(24):e042797; epub 2025-12-11; issue 2025-12-16; checked 2026-11-18",
        "url": "https://pubmed.ncbi.nlm.nih.gov/41378509/",
        "doi_url": "https://doi.org/10.1161/JAHA.125.042797",
        "current_check": "PubMed abstract and linked free PMC article read; no newer correction was found in the record checked on 2026-11-18.",
        "population": "Adult participants in same-device, same-protocol attended/unattended automated office BP comparisons; this synthesis is not a hypertension diagnostic-threshold guideline.",
    },
]

SOURCE_BY_ID = {x["source_id"]: x for x in SOURCES}
EVIDENCE: list[dict] = []


def ev(eid: str, source_id: str, locator: str, text: str, topics: list[str], population: str | None = None):
    src = SOURCE_BY_ID[source_id]
    EVIDENCE.append(
        {
            "id": eid,
            "source_id": source_id,
            "authority": src["authority"],
            "title": src["title"],
            "version": src["version"],
            "locator": locator,
            "url": src["url"],
            "kind": "review" if "_REVIEW_" in source_id else "guideline",
            "population": population or src["population"],
            "text": text,
            "topics": topics,
        }
    )


# NICE NG136: measurement/diagnosis; each item is a Chinese paraphrase of one
# recommendation, table, definition, or explicit condition.
ev("NICE-POP-01", "NICE_NG136", "Overview; recommendations, introductory scope", "NG136适用于18岁及以上成人；测量和诊断建议也适用于2型糖尿病成人。妊娠高血压另有独立指南。", ["适用人群", "妊娠边界"])
ev("NICE-MEAS-01", "NICE_NG136", "Recommendations 1.1.1", "测量血压的医务人员应接受充分的初始培训，并定期复核操作表现。", ["测量人员", "培训"])
ev("NICE-MEAS-02", "NICE_NG136", "Recommendations 1.1.2", "自动设备可能受脉搏不规则影响而不准确；测量前先触诊桡动脉或肱动脉脉搏。若脉搏不规则，改用肱动脉直接听诊的人工测量。", ["心律不齐", "人工听诊", "设备选择"])
ev("NICE-MEAS-03", "NICE_NG136", "Recommendations 1.1.3", "血压设备应经适当验证、维护，并按制造商说明定期重新校准。", ["设备验证", "维护", "校准"])
ev("NICE-MEAS-04", "NICE_NG136", "Recommendations 1.1.4", "诊室或家中测量应在安静、舒适温度的环境进行；受测者安静坐位，手臂伸展并有支撑，袖带尺寸须适合臂围。", ["标准化测量", "袖带", "姿势"])
ev("NICE-POST-01", "NICE_NG136", "Recommendations 1.1.5", "有体位性低血压症状（包括跌倒或体位性头晕）者，先仰卧测量；不便仰卧时可坐位测量，再于站立至少1分钟后复测。", ["体位性低血压", "测量时序", "例外条件"])
ev("NICE-POST-02", "NICE_NG136", "Recommendations 1.1.6", "站立至少1分钟后，收缩压下降至少20 mmHg或舒张压下降至少10 mmHg，即达到该建议的显著体位性下降阈值；后续血压改为站立位测量。", ["体位性低血压", "阈值", "后续测量"])
ev("NICE-POST-03", "NICE_NG136", "Recommendations 1.1.7–1.1.8", "症状提示体位性低血压而坐位起始的测量未达到阈值时，改从仰卧位重新测量；仍未证实但症状提示持续时，可转进一步专科评估。", ["体位性低血压", "阴性测量", "例外条件"])
ev("NICE-POST-04", "NICE_NG136", "Recommendations 1.4.16", "已诊断高血压者若合并2型糖尿病、体位性低血压症状或年龄≥80岁，应检查体位性低血压。", ["体位性低血压", "筛查人群", "年龄", "糖尿病"])
ev("NICE-DX-01", "NICE_NG136", "Recommendations 1.2.1", "考虑高血压诊断时，先测量双臂血压；若两臂读数相差超过15 mmHg，应重复测量。", ["双臂测量", "臂间差", "重复测量"])
ev("NICE-DX-02", "NICE_NG136", "Recommendations 1.2.1", "重复后两臂差异仍超过15 mmHg，后续测量使用读数较高的一臂。", ["臂间差", "后续测量"])
ev("NICE-DX-03", "NICE_NG136", "Recommendations 1.2.2", "诊室血压≥140/90 mmHg时，在当次咨询中再测一次；若第二次与第一次差异显著，再测第三次；诊室记录值取最后两次中的较低值。", ["诊室复测", "阈值", "取值规则"])
ev("NICE-DX-04", "NICE_NG136", "Recommendations 1.2.3", "诊室血压处于140/90至180/120 mmHg区间时，提供动态血压监测（ABPM）以确认诊断；达到严重血压范围时另按1.5节分流。", ["诊断确认", "ABPM", "范围"])
ev("NICE-DX-05", "NICE_NG136", "Recommendations 1.2.4", "若ABPM不合适或受测者不能耐受，可提供家庭血压监测（HBPM）确认诊断。", ["诊断确认", "HBPM", "ABPM不耐受"])
ev("NICE-DX-06", "NICE_NG136", "Recommendations 1.2.5", "等待诊断确认期间，先检查靶器官损害，再进行正式心血管风险评估。", ["等待确认", "靶器官损害", "风险评估"])
ev("NICE-ABPM-01", "NICE_NG136", "Recommendations 1.2.6", "ABPM确认时，通常清醒时段每小时至少取得2次测量，并以通常清醒时段至少14次读数的平均值确认。", ["ABPM方案", "清醒时段", "最少读数"])
ev("NICE-HBPM-01", "NICE_NG136", "Recommendations 1.2.7", "HBPM每次记录做连续两次测量，间隔至少1分钟且受测者坐位；每天测两次，理想安排为早晨和晚上。", ["HBPM方案", "重复测量", "时间"])
ev("NICE-HBPM-02", "NICE_NG136", "Recommendations 1.2.7", "HBPM至少记录4天、理想为7天；弃去第1天读数，以其余所有读数的平均值确认诊断。", ["HBPM方案", "天数", "首日处理", "平均值"])
ev("NICE-DX-07", "NICE_NG136", "Recommendations 1.2.8", "NICE确认高血压的组合阈值为：诊室≥140/90 mmHg，且ABPM清醒时段平均值或HBPM平均值≥135/85 mmHg。", ["诊断阈值", "诊室", "ABPM", "HBPM"])
ev("NICE-DX-08", "NICE_NG136", "Recommendations 1.2.9", "未达到高血压诊断，但已有靶器官损害证据时，可调查该损害的其他可能原因。", ["鉴别诊断", "靶器官损害", "未确诊"])
ev("NICE-DX-09", "NICE_NG136", "Recommendations 1.2.10", "未诊断高血压者之后至少每5年复测诊室血压；若低于140/90 mmHg但仍偏高，可缩短复测间隔。", ["复筛间隔", "阴性结果", "风险"])
ev("NICE-MONITOR-01", "NICE_NG136", "Recommendations 1.4.22", "对成人高血压患者用ABPM或HBPM监测反应时，采用其通常清醒时段的平均血压。", ["随访监测", "ABPM", "HBPM", "时段均值"])
ev("NICE-CLASS-01", "NICE_NG136", "Terms used; Stage 1 hypertension", "NICE第1期标准同时要求：诊室140/90至159/99 mmHg，且ABPM清醒均值或HBPM均值135/85至149/94 mmHg。", ["NICE分类", "第1期", "诊断阈值"])
ev("NICE-CLASS-02", "NICE_NG136", "Terms used; Stage 2 hypertension", "NICE第2期标准同时要求：诊室≥160/100且<180/120 mmHg，且ABPM清醒均值或HBPM均值≥150/95 mmHg。", ["NICE分类", "第2期", "诊断阈值"])
ev("NICE-URGENT-01", "NICE_NG136", "Recommendations 1.5.1", "诊室血压≥180/120 mmHg但没有当日转诊体征/症状时，应尽快检查靶器官损害；未见损害时，可在7天内复测诊室血压，或做ABPM/HBPM并确保7天内临床复核。", ["严重血压", "7天", "靶器官损害", "确认"])
ev("NICE-URGENT-02", "NICE_NG136", "Recommendations 1.5.2", "诊室血压≥180/120 mmHg并有视网膜出血、视乳头水肿，或新发意识混乱、胸痛、心衰征象、急性肾损伤等危及生命表现时，当日转专科评估。", ["紧急例外", "严重血压", "当日转诊"])
ev("NICE-URGENT-03", "NICE_NG136", "Recommendations 1.5.3", "疑似嗜铬细胞瘤时应当日专科评估；举例症状包括血压易变或体位性低血压、头痛、心悸、苍白、腹痛或出汗。", ["紧急例外", "嗜铬细胞瘤", "当日转诊"])

# 2025 AHA/ACC: keep its US classification and mapping separate from NICE/ISH.
ev("AHA-CLASS-01", "AHA_ACC_2025", "Section 2; Table 4", "AHA/ACC将正常血压列为收缩压<120且舒张压<80 mmHg。", ["AHA分类", "正常"])
ev("AHA-CLASS-02", "AHA_ACC_2025", "Section 2; Table 4", "AHA/ACC elevated BP为收缩压120–129 mmHg且舒张压<80 mmHg。", ["AHA分类", "升高血压"])
ev("AHA-CLASS-03", "AHA_ACC_2025", "Section 2; Table 4", "AHA/ACC第1期高血压为收缩压130–139 mmHg或舒张压80–89 mmHg。", ["AHA分类", "第1期", "阈值"])
ev("AHA-CLASS-04", "AHA_ACC_2025", "Section 2; Table 4", "AHA/ACC第2期高血压为收缩压≥140 mmHg或舒张压≥90 mmHg。", ["AHA分类", "第2期", "阈值"])
ev("AHA-CLASS-05", "AHA_ACC_2025", "Section 2; Table 4 footnote; Section 3", "AHA/ACC血压类别按至少2次谨慎测量的平均值，并以至少2个不同场合取得的读数为基础；单次读数不能单独代表该类别。", ["AHA分类", "测量次数", "多次就诊"])
ev("AHA-MEAS-01", "AHA_ACC_2025", "Section 3.1.1 Accurate Measurement of In-Office BP", "振荡法设备只有在以严格标准化参考方案验证准确度后，才属于建议使用的validated device。", ["设备验证", "振荡法", "诊室测量"])
ev("AHA-MEAS-02", "AHA_ACC_2025", "Section 3.1.1; Accurate Measurement of In-Office BP", "AHA/ACC要求测量人员接受初始和持续培训；理想情况下每6–12个月进行一次能力复核。", ["测量人员", "培训", "能力复核"])
ev("AHA-OUT-01", "AHA_ACC_2025", "Section 3.1.3 Out-of-Office BP Monitoring", "对疑似高血压成人，ABPM或HBPM等诊室外测量可提供诊室血压以外的信息，并用于确认诊断。", ["诊断确认", "诊室外测量", "ABPM", "HBPM"])
ev("AHA-THRESH-01", "AHA_ACC_2025", "Section 3.1.3; Table 7", "AHA/ACC表7中，诊室130/80 mmHg对应的HBPM和白天ABPM阈值为130/80，夜间ABPM为110/65，24小时ABPM为125/75 mmHg。", ["AHA阈值映射", "诊室外阈值", "第1期"])
ev("AHA-THRESH-02", "AHA_ACC_2025", "Section 3.1.3; Table 7", "AHA/ACC表7中，诊室140/90 mmHg对应的HBPM和白天ABPM阈值为135/85，夜间ABPM为120/70，24小时ABPM为130/80 mmHg。", ["AHA阈值映射", "诊室外阈值", "第2期"])
ev("AHA-WCH-01", "AHA_ACC_2025", "Section 3.2.2; Recommendation-Specific Supportive Text", "对诊室血压≥130/80 mmHg的成人，可用诊室外血压监测排除白大衣高血压；当诊室血压≥160/100时，指南另有及时处理注意事项。", ["白大衣高血压", "诊室外测量", "例外条件"])
ev("AHA-SCOPE-01", "AHA_ACC_2025", "Section 1.4 Scope of the Guideline; Table 2", "2025 AHA/ACC成人指南不提供慢性冠心病、心衰、心房颤动、瓣膜病、主动脉病或外周动脉病人群的相应管理建议；这些主题由其他专门指南覆盖。", ["适用范围", "心房颤动", "共病指南"])
ev("AHA-CUFFLESS-01", "AHA_ACC_2025", "Section 3.1.4; Top Take-Home Message 9", "在无袖带设备（包括智能手表）显示更高精度和可靠性前，不应依靠它们取得准确血压读数。", ["设备边界", "无袖带设备", "智能手表"])
ev("AHA-HOME-01", "AHA_ACC_2025", "Section 3.1.4; Home Blood Pressure Monitoring protocol", "家庭测量需用准确性经验证且袖带尺寸合适的血压设备；测量前30分钟避免吸烟、含咖啡因饮品和运动。", ["家庭测量", "设备验证", "测量准备"])
ev("AHA-HOME-02", "AHA_ACC_2025", "Section 3.1.4; Home Blood Pressure Monitoring protocol", "家庭测量时受测者坐椅、背部有支撑、双脚落地且不交叉，裸露手臂支撑在心脏高度，至少安静休息5分钟，测量期间不交谈或看手机/电视。", ["家庭测量", "姿势", "休息"])
ev("AHA-HOME-03", "AHA_ACC_2025", "Section 3.1.4; Home Blood Pressure Monitoring protocol", "AHA/ACC家庭监测方案每次间隔1分钟测2次、每日早晚各一组；早晨读数在排空膀胱后、服药和进食前取得。", ["家庭测量", "读数次数", "时间"])

# Latest global ISH guideline; its 130/80 elevated category is not merged with AHA or NICE.
ev("ISH-CLASS-01", "ISH_GLOBAL_2026", "Section 2; Table 3", "ISH 2026成人标准化诊室血压的正常类别为收缩压<130且舒张压<80 mmHg。", ["ISH分类", "正常"])
ev("ISH-CLASS-02", "ISH_GLOBAL_2026", "Section 2; Table 3", "ISH 2026将收缩压130–139 mmHg和/或舒张压80–89 mmHg列为Elevated BP。", ["ISH分类", "升高血压", "阈值"])
ev("ISH-CLASS-03", "ISH_GLOBAL_2026", "Section 2; Table 3", "ISH 2026第1级高血压为收缩压140–159 mmHg和/或舒张压90–99 mmHg。", ["ISH分类", "第1级", "阈值"])
ev("ISH-CLASS-04", "ISH_GLOBAL_2026", "Section 2; Table 3", "ISH 2026第2级高血压为收缩压≥160 mmHg和/或舒张压≥100 mmHg。", ["ISH分类", "第2级", "阈值"])
ev("ISH-CLASS-05", "ISH_GLOBAL_2026", "Section 2; Table 3 footnote", "ISH 2026分类参考至少2次就诊的平均血压；每次就诊取3次读数中的后2次。", ["ISH分类", "平均值", "读数规则"])
ev("ISH-DX-01", "ISH_GLOBAL_2026", "Section 2 Definition of Hypertension", "ISH 2026以标准化诊室血压≥140/90 mmHg（收缩压和/或舒张压）作为高血压阈值；须至少2次就诊确认，每次取3次读数。", ["ISH诊断", "确认阈值", "多次就诊"])
ev("ISH-DX-02", "ISH_GLOBAL_2026", "Section 2 Definition of Hypertension; Section 3.2", "ISH 2026建议尽可能以标准化HBPM或24小时ABPM确认诊室升高；若无法进行，可用后续就诊的重复诊室测量确认。", ["诊断确认", "HBPM", "ABPM", "替代路径"])
ev("ISH-DX-03", "ISH_GLOBAL_2026", "Section 3.2 Recommendations", "通常需2–3次诊室就诊确认高血压，间隔1–4周；就诊数取决于血压水平和总体心血管风险。", ["诊断确认", "就诊间隔", "例外条件"])
ev("ISH-DX-04", "ISH_GLOBAL_2026", "Section 3.2 Recommendations", "ISH 2026不以一次就诊直接诊断；例外为血压≥180/110 mmHg且已有高血压介导靶器官损害证据。", ["诊断例外", "严重血压", "靶器官损害"])
ev("ISH-MEAS-01", "ISH_GLOBAL_2026", "Section 3.2; Initial evaluation", "初次评估应测双臂；重复测量仍有一致的收缩压臂间差>10 mmHg时，后续使用读数较高的一臂；差异>20 mmHg可考虑进一步检查。", ["双臂测量", "臂间差", "复测"])
ev("ISH-POST-01", "ISH_GLOBAL_2026", "Section 3.2; Standing blood pressure", "站立血压不要求每位成人每次就诊都测。已治疗者有提示体位性低血压的症状时应测，尤其是>65岁、糖尿病或神经退行性疾病患者；站立3分钟内收缩压降≥20或舒张压降≥10 mmHg为体位性低血压。", ["体位性低血压", "人群例外", "阈值"])
ev("ISH-OUT-01", "ISH_GLOBAL_2026", "Section 3.3 Home and ambulatory BP monitoring", "ISH 2026认为多数情况下诊断需诊室外血压；130–159 mmHg收缩压和/或80–99 mmHg舒张压者尤其应以标准化HBPM或ABPM确认。", ["诊断确认", "范围", "HBPM", "ABPM"])
ev("ISH-OUT-02", "ISH_GLOBAL_2026", "Section 3.5 Other methods for out-of-office blood pressure monitoring", "ISH 2026称药房自动测得的血压可用于筛查与随访；若用于诊断，约135/85 mmHg为参考阈值，但要视测量条件和方案而定。", ["药房测量", "诊断阈值", "条件限制"])
ev("ISH-OUT-03", "ISH_GLOBAL_2026", "Section 3.5; Kiosk blood pressure monitors in public spaces", "ISH 2026认为公共场所自助亭式血压测量可用于人群筛查或偶尔监测，但不建议用于高血压诊断或管理。", ["公共亭式测量", "适用范围", "设备边界"])
ev("ISH-HOME-01", "ISH_GLOBAL_2026", "Section 3.3; Standardised Home BP Monitoring", "标准化家庭血压方案使用经过验证的上臂袖带设备；袖带式腕表或无袖带设备不作为该标准方案的替代。", ["家庭测量", "设备验证", "设备边界"])
ev("ISH-HOME-02", "ISH_GLOBAL_2026", "Section 3.3; Standardised Home BP Monitoring", "ISH家庭监测建议在连续7天内早晚各测一组，每组重复读数；首日读数不纳入平均。少于3天（12次读数）不足以形成该指南建议的家庭均值。", ["家庭测量", "7天", "首日处理", "最少读数"])
ev("ISH-HOME-03", "ISH_GLOBAL_2026", "Section 3.3; Standardised Home BP Monitoring", "ISH家庭测量诊断阈值为平均收缩压≥135 mmHg和/或平均舒张压≥85 mmHg。", ["家庭测量", "诊断阈值"])
ev("ISH-MEAS-02", "ISH_GLOBAL_2026", "Section 3.2; Standardised office BP measurement, Figure 3", "标准化诊室测量要求：安静室温舒适环境坐位休息5分钟、背部支撑、双脚平放不交叉、安静不说话；排空膀胱，测量前30分钟不饮咖啡因饮品。", ["标准化测量", "测量准备", "姿势"])
ev("ISH-MEAS-03", "ISH_GLOBAL_2026", "Section 3.2; Standardised office BP measurement, Figure 3", "袖带置于有支撑、与心脏同高的上臂；每次就诊测2–3次、间隔1分钟，取最后两次稳定读数的平均；优先用验证过的上臂电子设备。", ["标准化测量", "袖带", "重复测量", "设备"])
ev("ISH-MEAS-04", "ISH_GLOBAL_2026", "Section 3.2; Standardised office BP measurement, Figure 3", "若无经验证电子设备，ISH允许使用质量良好的防震人工听诊血压计；心房颤动等心律失常时优先人工听诊。", ["心律不齐", "人工听诊", "资源例外"])

# USPSTF 2021 final recommendation: a screening recommendation, not a diagnostic cutoff guideline.
ev("USP-POP-01", "USPSTF_SCREEN_2021", "Recommendation Summary; Patient Population Under Consideration", "USPSTF 2021筛查建议对象为≥18岁且既往未确诊高血压的成人。", ["适用人群", "筛查"])
ev("USP-SCREEN-01", "USPSTF_SCREEN_2021", "Recommendation Summary; Screening Tests", "USPSTF建议成人先用诊室血压测量筛查；若阳性，在开始治疗前用诊室外测量确认诊断。", ["筛查流程", "诊室测量", "诊断确认"])
ev("USP-ABPM-01", "USPSTF_SCREEN_2021", "Clinician Summary; Screening Tests", "USPSTF描述的ABPM使用可编程便携设备，通常在日常活动或睡眠期间每20–30分钟自动测量，持续12–24小时。", ["ABPM", "监测时长", "间隔"])
ev("USP-HBPM-01", "USPSTF_SCREEN_2021", "Clinician Summary; Screening Tests", "USPSTF描述HBPM为用自动设备在家自测，频率低于ABPM，例如每日或每周1–2次；读数可跨更长时段收集。", ["HBPM", "测量频率"])
ev("USP-PREP-01", "USPSTF_SCREEN_2021", "Clinician Summary; Screening Tests", "筛查测量应使用经验证且准确的设备，在上臂肱动脉处、坐位安静休息5分钟后完成。", ["测量准备", "设备", "姿势"])
ev("USP-INTERVAL-01", "USPSTF_SCREEN_2021", "Clinician Summary; Screening Intervals", "USPSTF称最佳筛查间隔证据有限；合理选项包括≥40岁或高风险成人每年筛查，18–39岁、无高风险且先前读数正常者每3–5年筛查。", ["筛查间隔", "年龄", "风险"])
ev("USP-BOUNDARY-01", "USPSTF_SCREEN_2021", "Recommendation Summary; Definitions", "这项USPSTF建议是成人筛查建议，所审研究采用的高血压定义阈值不一；它没有规定一个适用于所有机构的统一诊断阈值。", ["证据边界", "阈值差异", "筛查"])

# USPSTF evidence report: numerical findings stay explicitly attached to this review.
ev("REV-SCOPE-01", "USPSTF_REVIEW_2021", "Abstract; Methods—Study Selection", "系统综述评估成人高血压筛查与确诊测量；纳入未治疗成人或按治疗状态分层研究，确认准确性研究以ABPM为参考标准。检索监测截至2021-03-26。", ["综述范围", "人群", "ABPM参考标准", "日期"])
ev("REV-SIZE-01", "USPSTF_REVIEW_2021", "Abstract—Results", "综述共纳入52项研究、215,534人；这是筛查准确性和伤害的证据综述规模，不是单一诊断队列。", ["综述样本", "研究规模", "证据类型"])
ev("REV-OFFICE-01", "USPSTF_REVIEW_2021", "Abstract—Results; initial office-based screening", "15项研究（n=11,309）初筛诊室测量的汇总敏感度为0.54（95% CI 0.37–0.70），特异度为0.90（0.84–0.95）；研究间存在显著临床和统计异质性。", ["系统综述结果", "诊室筛查", "敏感度", "特异度"])
ev("REV-OFFICE-CONFIRM-01", "USPSTF_REVIEW_2021", "Abstract—Results; office-based confirmation", "8项诊室确诊测量研究（n=53,183）的汇总敏感度为0.80（95% CI 0.68–0.88），特异度为0.55（0.42–0.66）。", ["系统综述结果", "诊室确诊", "敏感度", "特异度"])
ev("REV-HOME-01", "USPSTF_REVIEW_2021", "Abstract—Results; home-based confirmation", "4项家庭确诊测量研究（n=1,001）的汇总敏感度为0.84（95% CI 0.76–0.90），特异度为0.60（0.48–0.71）。", ["系统综述结果", "家庭确诊", "敏感度", "特异度"])
ev("REV-HETEROGENEITY-01", "USPSTF_REVIEW_2021", "Abstract—Results; confirmatory modalities", "18项确诊测量研究（n=57,128）涵盖多种方式，整体结果异质；准确性数字应按综述所纳入的特定方式和人群解读。", ["系统综述结果", "异质性", "方法比较"])
ev("REV-HARM-01", "USPSTF_REVIEW_2021", "Abstract—Results; screening harms", "ABPM与暂时性睡眠障碍和瘀青相关；13项研究（n=5,150）提示筛查与生活质量下降或心理困扰无关联，旷工证据不一致。", ["系统综述结果", "伤害", "ABPM"])
ev("JAHA-AOBP-01", "JAHA_AOBP_REVIEW_2025", "Abstract—Methods and Results", "该系统综述纳入15项同参与者的配对研究（n=1,747）；有人与无人自动诊室测量按随机或交替顺序，在同一就诊中使用同一设备和测量方案比较。", ["系统综述", "自动诊室测量", "配对设计", "研究规模"])
ev("JAHA-AOBP-02", "JAHA_AOBP_REVIEW_2025", "Abstract—Results; systolic and diastolic differences", "全样本中，无人测量较有人测量的收缩压平均低2.7 mmHg（95% CI −4.7至−0.6），舒张压低0.9 mmHg（−1.8至−0.1）；排除单个离群研究后，收缩压差为−2.0 mmHg（−4.1至0.0）。", ["系统综述", "测量者在场", "结果估计", "敏感性分析"])
ev("JAHA-AOBP-03", "JAHA_AOBP_REVIEW_2025", "Abstract—Conclusions", "该综述认为标准化无人自动诊室读数略低于有人读数，但这些小差值是否会改变临床管理尚不确定；有人自动诊室测量仍是可行选项。", ["系统综述", "临床解释", "不确定性"])


QUESTIONS: list[dict] = []


def q(
    qid: str,
    family: str,
    split: str,
    query: str,
    gold: list[str] | None = None,
    supplied: list[str] | None = None,
    groups: list[list[str]] | None = None,
    complexity: str = "single",
    reason: str = "",
    unanswerable: bool = False,
    scope_out: bool = False,
):
    gold = gold or []
    supplied = supplied or []
    groups = groups or []
    expected = [g[0] for g in groups]
    initial_need = False if scope_out else (unanswerable or not set(gold).issubset(set(supplied)))
    QUESTIONS.append(
        {
            "id": qid,
            "group": qid,
            "family": family,
            "split": split,
            "query": query,
            "supplied": supplied,
            "gold": gold,
            "expected": expected,
            "expected_groups": groups,
            "unanswerable": unanswerable,
            "complexity": complexity,
            "reason": reason,
            "scope_out": scope_out,
            "initial_need": initial_need,
        }
    )


REFUSAL = [["现有资料未规定", "本语料没有这项细节", "证据不足，不能给出该参数"]]
SCOPE_REFUSAL = [["超出本语料范围", "需要针对该人群的专门指南", "不能据此作个体用药建议"]]

# Development split: each row is a distinct intent group; all human grading labels stay here.
q("D01", "classification", "dev", "按2025 AHA/ACC成人指南，收缩压133、舒张压76 mmHg属于哪个类别？", ["AHA-CLASS-03"], [], [["第1期高血压", "Stage 1 hypertension"]], reason="由AHA收缩压区间判类；不套用其他机构阈值。")
q("D02", "measurement", "dev", "NICE要求血压设备满足哪些基本维护条件？", ["NICE-MEAS-03"], ["NICE-MEAS-04"], [["验证", "经验证"], ["定期校准", "按制造商说明重新校准"]], reason="正确上下文只给出环境和姿势，设备标准需检索。")
q("D03", "evidence_review", "dev", "相对ABPM参考标准，初筛诊室测量和家庭确诊测量的汇总敏感度、特异度各是多少？", ["REV-OFFICE-01", "REV-HOME-01"], [], [["初筛诊室：敏感度0.54、特异度0.90", "office screen 0.54/0.90"], ["家庭确诊：敏感度0.84、特异度0.60", "home confirmation 0.84/0.60"]], complexity="multi", reason="跨两个准确性分析分项作比较；数值只代表综述所纳入研究。")
q("D04", "measurement", "dev", "按NICE，哪些人群的高血压评估应检查体位性低血压？", ["NICE-POST-04"], ["NICE-POST-04"], [["2型糖尿病", "type 2 diabetes"], ["体位性低血压症状", "postural hypotension symptoms"], ["80岁及以上", "≥80岁"]], reason="全部依据已在可信上下文中，供给层可直接作答。")
q("D05", "measurement", "test", "有体位性头晕的成人，NICE建议如何安排起始和站立测量？", ["NICE-POST-01"], [], [["先仰卧（不便时坐位）", "先仰卧位"], ["站立至少1分钟后复测", "站立1分钟后复测"]], reason="体位症状触发的测量时序。")
q("D06", "measurement", "test", "NICE判定显著体位性血压下降的两个阈值是什么？", ["NICE-POST-02"], ["NICE-POST-01"], [["收缩压下降≥20 mmHg", "收缩压下降20 mmHg或更多"], ["舒张压下降≥10 mmHg", "舒张压下降10 mmHg或更多"]], reason="已给出体位测量方法，阈值仍需检索。")
q("D07", "confirmation", "dev", "NICE在等待高血压确诊时，先后安排哪两项评估？", ["NICE-DX-06"], ["NICE-DX-04"], [["先检查靶器官损害", "target organ damage first"], ["再正式评估心血管风险", "then formal cardiovascular risk"]], reason="上下文仅给确诊监测路径，不含等待期间的评估顺序。")
q("D08", "confirmation", "dev", "NICE何时优先提供ABPM，何种情况下可改用HBPM？", ["NICE-DX-04", "NICE-DX-05"], [], [["诊室140/90至180/120 mmHg提供ABPM", "140/90–180/120时ABPM"], ["ABPM不合适或不能耐受时HBPM", "不能耐受ABPM时用HBPM"]], complexity="multi", reason="需同时给出区间和替代条件。")
q("D09", "confirmation", "dev", "NICE用多少个清醒时段ABPM读数确认诊断，且每小时至少几次？", ["NICE-ABPM-01"], ["NICE-DX-04"], [["至少14次读数的平均值", "≥14次读数平均"], ["每小时至少2次", "≥2次/小时"]], reason="已提供转入ABPM的条件，具体有效读数标准缺失。")
q("D10", "confirmation", "dev", "未达到NICE高血压诊断但存在靶器官损害时，后续需要考虑什么？", ["NICE-DX-08"], [], [["调查靶器官损害的其他可能原因", "investigate alternative causes"]], reason="诊断阴性而器官损害存在时的鉴别方向。")
q("D11", "confirmation", "dev", "对NICE家庭确诊读数，每次测几次、一天几次，间隔多久？", ["NICE-HBPM-01"], [], [["每次连续2次", "每次两次"], ["间隔至少1分钟", "至少间隔1分钟"], ["每天2次，早晚为宜", "每日早晚各一次"]], reason="单条证据覆盖三项相互配合的程序条件。")
q("D12", "classification", "dev", "NICE确诊标准如何组合诊室与诊室外数值？", ["NICE-DX-07"], ["NICE-DX-03"], [["诊室≥140/90 mmHg", "诊室140/90或以上"], ["ABPM清醒均值或HBPM均值≥135/85 mmHg", "诊室外平均135/85或以上"]], reason="诊室读数阈值已给，需补齐配对诊断阈值。")
q("D13", "classification", "dev", "NICE第1期与第2期诊断分别要求什么诊室外均值？", ["NICE-CLASS-01", "NICE-CLASS-02"], [], [["第1期：135/85至149/94 mmHg", "Stage 1: 135/85–149/94"], ["第2期：≥150/95 mmHg", "Stage 2: 150/95或以上"]], complexity="multi", reason="比较两个连续分期，需分别给出不同的联合范围。")
q("D14", "screening", "dev", "USPSTF证据综述覆盖的机构高血压定义阈值范围是多少？该建议是否自行规定统一界值？", ["USP-BOUNDARY-01"], [], [["各机构约从130/80到140/90 mmHg", "130/80至140/90"], ["没有给出统一诊断阈值", "未规定统一cutoff"]], reason="区分筛查建议与各机构诊断定义。")
q("D15", "screening", "dev", "USPSTF所说的较低风险18–39岁成人，合理复筛间隔是什么？", ["USP-INTERVAL-01"], [], [["每3–5年", "3至5年一次"]], reason="年龄、风险及既往正常读数构成该间隔适用条件。")
q("D16", "screening", "dev", "USPSTF建议什么年龄或风险条件下每年筛查？", ["USP-INTERVAL-01"], ["USP-POP-01"], [["≥40岁", "40岁及以上"], ["高风险成人", "风险升高者"]], reason="已给推荐人口范围，需检索年龄/风险频率。")
q("D17", "measurement", "dev", "按USPSTF摘要，筛查时设备、部位、姿势与休息条件是什么？", ["USP-PREP-01"], ["USP-SCREEN-01"], [["经验证且准确的设备", "validated accurate device"], ["上臂肱动脉，坐位", "upper arm brachial artery, seated"], ["休息5分钟", "5 min rest"]], reason="已提供筛查流程，具体测量条件仍需补证。")
q("D18", "screening", "dev", "USPSTF描述的ABPM大致多久测一次、连续多久？", ["USP-ABPM-01"], [], [["每20–30分钟", "20至30分钟一次"], ["12–24小时", "持续12至24小时"]], reason="单一证据项包含频率与时长。")
q("D19", "evidence_review", "dev", "USPSTF系统综述的确诊准确性研究采用什么参考标准，通常纳入何种治疗状态成人？", ["REV-SCOPE-01"], [], [["ABPM参考标准", "ABPM作为reference standard"], ["未治疗成人或按治疗状态分层", "untreated or stratified by treatment"]], reason="考察综述纳入与分析设计，不把该证据当作诊断指南。")
q("D20", "evidence_review", "dev", "该系统综述的家庭确诊测量敏感度/特异度是多少，来自多少项研究？", ["REV-HOME-01"], ["REV-SIZE-01"], [["敏感度0.84", "84%"], ["特异度0.60", "60%"], ["4项研究", "4 studies"]], reason="需从已给的综述总规模上下文中补取对应分项。")
q("D21", "classification", "dev", "AHA/ACC成人第2期高血压的收缩压或舒张压界值是什么？", ["AHA-CLASS-04"], [], [["收缩压≥140 mmHg", "SBP 140或以上"], ["或舒张压≥90 mmHg", "DBP 90或以上"]], reason="单一分类项目，保留‘或’关系。")
q("D22", "classification", "dev", "AHA/ACC 120至129的收缩压在什么舒张压条件下属于elevated BP？", ["AHA-CLASS-02"], ["AHA-CLASS-01"], [["舒张压<80 mmHg", "DBP低于80"]], reason="上下文给出正常血压，只需补齐升高类别的舒张压约束。")
q("D23", "measurement", "dev", "AHA/ACC推荐的振荡式设备须满足什么验证条件？", ["AHA-MEAS-01"], [], [["经严格标准化参考方案验证准确度", "rigorous standardized validation"]], reason="设备验证规则；不涉及设备品牌。")
q("D24", "classification", "dev", "AHA/ACC第1期高血压的阈值是什么，且分类依据怎样的重复测量？", ["AHA-CLASS-03", "AHA-CLASS-05"], [], [["收缩压130–139或舒张压80–89 mmHg", "SBP 130–139 or DBP 80–89"], ["至少2次谨慎读数的平均，来自至少2个场合", "average ≥2 readings on ≥2 occasions"]], complexity="multi", reason="数值分类和重复测量基础由两个不同证据项支持。")
q("D25", "measurement", "dev", "AHA/ACC家庭测量前30分钟应避免哪些活动或摄入？", ["AHA-HOME-01"], ["AHA-HOME-02"], [["吸烟", "smoking"], ["含咖啡因饮品", "咖啡因"], ["运动", "exercise"]], reason="所给姿势证据不包含测前限制。")
q("D26", "classification", "dev", "ISH 2026成人标准化诊室血压的正常类别同时满足什么上限？", ["ISH-CLASS-01"], ["ISH-CLASS-02"], [["收缩压<130 mmHg", "SBP below 130"], ["舒张压<80 mmHg", "DBP below 80"]], reason="上下文给升高血压类别，问题询问较低的双条件类别。")
q("D27", "confirmation", "dev", "ISH 2026诊室收缩压138 mmHg、舒张压84 mmHg时建议用什么方式确认，且该值处于特别提示的范围吗？", ["ISH-OUT-01", "ISH-DX-02"], [], [["sHBPM或ABPM确认", "sHBPM or ABPM"], ["收缩压130–159和/或舒张压80–99范围", "within 130–159 and/or 80–99"]], complexity="multi", reason="需将读数映射至ISH确认区间并给出可用诊室外方法；不作治疗判断。")
q("D28", "confirmation", "dev", "ISH 2026诊断确认常用哪些诊室外测量？", ["ISH-DX-02"], ["ISH-DX-02"], [["标准化家庭血压监测", "sHBPM"], ["24小时动态血压监测", "24-h ABPM"]], reason="有效来源已完整提供，供给上下文可免额外检索。")
q("D29", "measurement", "dev", "ISH 2026一次标准化诊室测量应做几次，间隔多久，并如何汇总读数？", ["ISH-MEAS-03"], [], [["2–3次", "2–3 readings"], ["间隔1分钟", "1-minute intervals"], ["平均最后两次稳定读数", "average last two stable readings"]], reason="询问单次测量操作，不与诊断就诊次数混淆。")
q("D30", "measurement", "dev", "ISH如何解释药房自动测量的诊断阈值及其使用限制？", ["ISH-OUT-02"], [], [["约135/85 mmHg", "approximately 135/85"], ["取决于测量条件与方案", "depends on conditions and protocol"]], reason="药房读数有场景限制，阈值不是无条件通用值。")
q("D31", "measurement", "dev", "ISH 2026在房颤等心律失常且电子设备不适用时优先采用何种测量？", ["ISH-MEAS-04"], [], [["人工听诊", "manual auscultation"], ["有质量保证的防震血压计", "quality shock-resistant sphygmomanometer"]], reason="特殊节律与设备可用性触发方法例外。")
q("D32", "measurement", "dev", "ISH是否建议使用公共场所自助亭式读数诊断高血压？", ["ISH-OUT-03"], ["ISH-HOME-01"], [["不建议用于诊断或管理", "not recommended for diagnosis or management"], ["可用于人群筛查或偶尔监测", "screening or occasional monitoring only"]], reason="家庭设备上下文相关但不等价，问题指定公共自助设备。")
q("D33", "exceptions", "dev", "按NICE，成人诊室血压≥180/120但无当日转诊体征时，未发现靶器官损害后如何及时确认？", ["NICE-URGENT-01"], ["NICE-URGENT-02"], [["7天内复测诊室血压", "repeat clinic within 7 days"], ["或ABPM/HBPM并确保7天内临床复核", "or ABPM/HBPM plus review within 7 days"]], reason="提供了有红旗时的当日转诊上下文，需区分无红旗路径。")
q("D34", "exceptions", "dev", "NICE对疑似嗜铬细胞瘤的专科评估时限是什么？", ["NICE-URGENT-03"], [], [["当日", "同一天"]], reason="带症状例子的单一紧急条件。")
q("D35", "scope_boundary", "dev", "USPSTF成人筛查建议是否为17岁青少年提供血压诊断阈值？", ["USP-POP-01"], ["USP-POP-01"], [["不适用；对象为18岁及以上成人", "recommendation applies to adults ≥18"]], complexity="out", reason="该建议人口范围为成人，儿童/青少年需要相应人群来源。", unanswerable=True, scope_out=True)
q("D36", "evidence_gap", "dev", "NICE说第二次读数与首次‘差异显著’时测第三次，是否给出精确mmHg差值？", [], ["NICE-DX-03"], REFUSAL, complexity="out", reason="推荐正文没有规定数值阈值，不应自行设定。", unanswerable=True)
q("D37", "review_evidence", "dev", "在同设备、同方案的配对研究中，无人自动诊室测量相对有人测量的平均差值是多少？排除离群研究后有何变化？", ["JAHA-AOBP-01", "JAHA-AOBP-02"], ["ISH-MEAS-03"], [["15项研究、1,747人，同一就诊同设备同方案配对", "15 studies, n=1747, same-visit paired protocol"], ["全样本收缩压低2.7 mmHg", "SBP −2.7 mmHg"], ["排除离群研究后低2.0 mmHg，CI含0", "leave-one-out −2.0 mmHg, CI includes 0"]], complexity="multi", reason="将综述设计和估计值联合呈现；综述不形成新的诊断阈值。")
q("D38", "measurement", "dev", "AHA/ACC建议测量人员多久接受一次理想的血压测量能力复核？", ["AHA-MEAS-02"], [], [["每6–12个月", "every 6–12 months"]], reason="指南把培训能力复核作为测量质量条件。")

# Test split: distinct intents/groups from dev; includes empty, partial, full-valid,
# and irrelevant supplied context. These labels must never enter the retrieval KB.
q("T01", "source_comparison", "test", "AHA/ACC与ISH 2026分别把130–139/80–89 mmHg归为什么类别？", ["AHA-CLASS-03", "ISH-CLASS-02"], ["AHA-CLASS-03"], [["AHA/ACC第1期高血压", "AHA Stage 1 hypertension"], ["ISH Elevated BP", "ISH升高血压"]], complexity="multi", reason="同一范围在两指南中类别命名不同；不得混成一个阈值。")
q("T02", "source_comparison", "test", "按NICE与ISH 2026，首次双臂测量后，持续臂间收缩压差多少会改用高读数臂？", ["NICE-DX-01", "NICE-DX-02", "ISH-MEAS-01"], [], [["NICE持续差>15 mmHg", "NICE超过15 mmHg"], ["ISH一致差>10 mmHg", "ISH超过10 mmHg"], ["后续用较高读数臂", "higher arm for subsequent readings"]], complexity="multi", reason="需逐机构报告不同阈值；NICE要求复测后仍超15，ISH强调重复中一致>10。")
q("T03", "confirmation", "test", "NICE对诊室150/96 mmHg的成人如何复测，并用什么方式确认？", ["NICE-DX-03", "NICE-DX-04"], ["NICE-DX-03"], [["同次至少第二次；差异显著再测第三次", "second, then third if substantially different"], ["提供ABPM", "offer ABPM"]], complexity="multi", reason="给定证据只有诊室记录规则，确认方式仍缺。")
q("T04", "measurement", "test", "比较NICE与ISH家庭诊断方案：是否弃去第一天，分别记录至少几天？", ["NICE-HBPM-02", "ISH-HOME-02"], ["NICE-HBPM-02"], [["NICE弃首日；至少4天、理想7天", "NICE discards day 1; ≥4 ideally 7"], ["ISH弃首日；建议连续7天", "ISH discards day 1; 7 days"]], complexity="multi", reason="相似规则需分别归属来源，并比较最少天数。")
q("T05", "source_comparison", "test", "给医生一个AHA/ACC 140/90 mmHg诊室值对应的夜间和24小时ABPM表值。", ["AHA-THRESH-02"], ["AHA-THRESH-02"], [["夜间120/70 mmHg", "night 120/70"], ["24小时130/80 mmHg", "24-hour 130/80"]], reason="同一已校验上下文包含完整映射，可直接答，不需检索。")
q("T06", "classification", "test", "按ISH 2026的分级，成人标准化诊室160/100 mmHg处于哪个等级？", ["ISH-CLASS-04"], ["NICE-CLASS-02"], [["第2级高血压", "Grade 2 hypertension"]], reason="干扰上下文给NICE第2期外部均值，不能替代ISH的分类。")
q("T07", "confirmation", "test", "ISH 2026通常要求几次诊室就诊确认，且一般间隔几周？", ["ISH-DX-03"], [], [["2–3次", "two to three"], ["1–4周", "one to four weeks"]], reason="单项证据中的诊断时序。")
q("T08", "exceptions", "test", "ISH 2026何种情况允许不等待多次就诊便在一次就诊诊断？", ["ISH-DX-04"], ["ISH-DX-01"], [["血压≥180/110 mmHg且有靶器官损害证据", "≥180/110 plus HMOD"]], reason="已给普通确诊要求，问题询问有限例外。")
q("T09", "source_comparison", "test", "按NICE，诊室140/90至180/120 mmHg时提供什么确认监测？按ISH通常需几次就诊、间隔几周确认？", ["NICE-DX-04", "ISH-DX-03"], [], [["NICE在140/90至180/120区间提供ABPM", "NICE offers ABPM in range"], ["ISH通常2–3次就诊、间隔1–4周", "ISH 2–3 visits 1–4 weeks"]], complexity="multi", reason="两个机构的确认路径及时间尺度分别作答。")
q("T10", "measurement", "test", "按ISH标准化诊室方案，安静坐位准备多久，袖带臂应怎样放置？", ["ISH-MEAS-02", "ISH-MEAS-03"], ["ISH-MEAS-02"], [["休息5分钟", "5 min rest"], ["有支撑且中臂与心脏同高", "arm supported at heart level"]], complexity="multi", reason="已供应准备条件，臂位在另一条证据中。")
q("T11", "measurement", "test", "对出现体位性症状的成人，NICE与ISH分别从何种姿势开始、何时测站立血压？ISH的重点筛查人群又有哪些？", ["NICE-POST-01", "ISH-POST-01"], [], [["NICE先仰卧（不便时坐位），站立至少1分钟复测", "NICE supine or seated then stand ≥1 min"], ["ISH对已治疗且有症状者测量，特别关注>65岁、糖尿病或神经退行性疾病", "ISH treated symptomatic groups"], ["ISH站立后3分钟内，收缩压降≥20或舒张压降≥10 mmHg", "ISH within 3 min, drop ≥20/10 mmHg"]], complexity="multi", reason="联合比较体位方案、适用人群和测量时限。")
q("T12", "screening", "test", "USPSTF的筛查建议适用于哪类成人，何时须进行诊室外确认？", ["USP-POP-01", "USP-SCREEN-01"], [], [["≥18岁且未确诊高血压", "adults ≥18 without known hypertension"], ["阳性诊室筛查后、治疗前", "after positive office screen before treatment"]], complexity="multi", reason="需人口范围与确诊步骤两项。")
q("T13", "source_comparison", "test", "NICE与AHA/ACC的140/90诊室值分别对应怎样的诊断外部测量阈值？", ["NICE-DX-07", "AHA-THRESH-02"], ["NICE-DX-07"], [["NICE HBPM/清醒ABPM≥135/85", "NICE ≥135/85"], ["AHA/ACC HBPM/白天135/85、夜间120/70、24小时130/80", "AHA mapping 135/85, 120/70, 130/80"]], complexity="multi", reason="跨指南数值映射，NICE上下文不含AHA分时段阈值。")
q("T14", "evidence_review", "test", "综述里诊室确认测量的准确性结果，敏感度与特异度各多少？", ["REV-OFFICE-CONFIRM-01"], ["REV-OFFICE-01"], [["敏感度0.80", "80%"], ["特异度0.55", "55%"]], reason="供给为初筛准确率；需查询不同阶段的诊室确诊证据。")
q("T15", "evidence_review", "test", "USPSTF综述比较诊室初筛与诊室确诊时，二者的敏感度和特异度各是多少？", ["REV-OFFICE-01", "REV-OFFICE-CONFIRM-01"], ["REV-SIZE-01"], [["初筛：0.54敏感度、0.90特异度", "screening 0.54/0.90"], ["诊室确诊：0.80敏感度、0.55特异度", "office confirmation 0.80/0.55"]], complexity="multi", reason="同一综述中两类测试用途的数据来自不同分析分项，需联合读取。")
q("T16", "evidence_review", "test", "USPSTF证据综述观察到ABPM的哪些短期不适？", ["REV-HARM-01"], [], [["暂时性睡眠障碍", "temporary sleep disturbance"], ["瘀青", "bruising"]], reason="属于系统综述伤害结果，不作为筛查频率建议。")
q("T17", "scope_boundary", "test", "医生询问该成人语料能否决定某位心衰患者的具体降压药和剂量；系统应给什么答复？", [], ["NICE-POP-01"], SCOPE_REFUSAL, complexity="out", reason="个体用药属于明确排除的处方服务；资料不提供个体药物方案。", unanswerable=True, scope_out=True)
q("T18", "evidence_gap", "test", "现有指南是否给出测量上臂围52 cm成人的具体袖带品牌和制造商货号？", [], ["ISH-MEAS-03"], REFUSAL, complexity="out", reason="来源要求正确袖带但本语料没有品牌/货号，不能编造。", unanswerable=True)
q("T19", "exceptions", "test", "NICE诊室读数180/120且合并新发意识混乱，应如何分流？", ["NICE-URGENT-02"], [], [["当日专科评估", "同日转诊"], ["严重血压并有危及生命症状", "≥180/120 with life-threatening symptoms"]], complexity="single", reason="红旗症状触发急诊式专科分流，不继续常规确认题。")
q("T20", "measurement", "test", "NICE对成人高血压患者用ABPM或HBPM监测反应时，应取哪个时段的平均值？", ["NICE-MONITOR-01"], ["NICE-MONITOR-01"], [["通常清醒时段平均值", "usual waking-hours average"]], reason="可信上下文已含答案，可免额外检索。")
q("T21", "classification", "test", "2025 AHA/ACC规定的第1期阈值是否以收缩压和舒张压同时超标为条件？", ["AHA-CLASS-03"], ["ISH-CLASS-03"], [["否；收缩压130–139或舒张压80–89即可", "任一达到即可"]], reason="不相关ISH第1级资料可能诱发阈值混淆；保留AHA的‘或’。")
q("T22", "source_comparison", "test", "NICE与ISH对未诊断成人的证据性家庭阈值是否相同？请分别给出收缩压或舒张压界值。", ["NICE-DX-07", "ISH-HOME-03"], [], [["两者均为平均收缩压≥135 mmHg或舒张压≥85 mmHg", "both ≥135/85"]], complexity="multi", reason="共同数值须由两套各自规则共同支持，不能据相似而省去任一来源。")
q("T23", "evidence_gap", "test", "NICE是否为跨越时区的旅行者规定了ABPM监测的换算时区表？", [], ["NICE-ABPM-01"], REFUSAL, complexity="out", reason="语料提供通常清醒时段，但没有跨时区专门算法，属于证据缺口。", unanswerable=True)
q("T24", "scope_boundary", "test", "某位孕妇的血压诊断及用药应按本语料中的成人阈值直接决定吗？", ["NICE-POP-01"], ["NICE-POP-01"], [["不应；需查妊娠专门指南", "pregnancy requires dedicated guideline"]], complexity="out", reason="人口范围明确排除妊娠；且请求涉及个体治疗，需专门来源与临床评估。", unanswerable=True, scope_out=True)


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8", newline="\n")


def main():
    DATA = OUT
    write_jsonl(DATA / "evidence.jsonl", EVIDENCE)
    write_jsonl(DATA / "questions.jsonl", QUESTIONS)
    (DATA / "sources.json").write_text(json.dumps(SOURCES, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"evidence={len(EVIDENCE)} questions={len(QUESTIONS)} dev={sum(q['split']=='dev' for q in QUESTIONS)} test={sum(q['split']=='test' for q in QUESTIONS)}")


if __name__ == "__main__":
    main()
