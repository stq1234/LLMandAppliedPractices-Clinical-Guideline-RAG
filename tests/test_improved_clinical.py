from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cli import _build_parser
from improved_clinical import System, _explicit_facets


class StubBackend:
    embedding_signature = "improved-clinical-unit-stub-v1"

    def __init__(self):
        self.chat_calls = 0
        self.bad_planner = False

    @staticmethod
    def _vector(text: str, dims: int = 96) -> np.ndarray:
        values = np.zeros(dims, dtype=np.float32)
        normalized = str(text).casefold()
        features = set()
        for word in normalized.split():
            features.add(word)
        cjk = "".join(ch for ch in normalized if "\u3400" <= ch <= "\u9fff")
        features.update(cjk[i:i + 2] for i in range(max(0, len(cjk) - 1)))
        for feature in features:
            slot = sum((i + 1) * ord(ch) for i, ch in enumerate(feature)) % dims
            values[slot] += 1.0
        norm = float(np.linalg.norm(values))
        return values / norm if norm else values

    def encode(self, texts, query=False):
        return np.stack([self._vector(text) for text in texts])

    def chat(self, system: str, user: str) -> str:
        self.chat_calls += 1
        payload = json.loads(user)
        if "faithful evidence-query planner" in system:
            query = payload["question"]
            if self.bad_planner:
                facets = [query.split("；")[0], "推荐增加治疗管理建议"]
            else:
                facets = _explicit_facets(query)
            return json.dumps({"facets": facets}, ensure_ascii=False)

        if "evidence checker" in system:
            results = []
            sources = payload.get("sources", [])
            for need in payload.get("needs", []):
                facet = str(need["text"])
                if "第一天" in facet or "首日" in facet:
                    results.append({"index": need["index"], "supported": False,
                                    "citations": [], "followup_query": "unsupported invented query"})
                    continue
                source = sources[0] if sources else None
                if source:
                    results.append({"index": need["index"], "supported": True,
                                    "citations": [{"id": source["id"], "quote": source["text"]}],
                                    "followup_query": "invented management advice"})
                else:
                    results.append({"index": need["index"], "supported": False,
                                    "citations": [], "followup_query": "invented management advice"})
            return json.dumps({"facets": results}, ensure_ascii=False)

        if "Check whether each factual claim" in system:
            claims = []
            for item in payload.get("claims", []):
                ids = [citation["id"] for citation in item.get("citations", [])]
                claims.append({"index": item["index"], "supported": bool(ids), "ids": ids})
            return json.dumps({"claims": claims}, ensure_ascii=False)

        if "Answer only" in system:
            sources = payload.get("sources", [])
            if not sources:
                return json.dumps({"claims": []})
            source = sources[0]
            return json.dumps({
                "claims": [{
                    "text": "成人测量应使用经过验证的设备。",
                    "citations": [{"id": source["id"], "quote": source["text"]}],
                }]
            }, ensure_ascii=False)

        raise AssertionError("Unexpected model prompt in test stub")


def _corpus() -> list[dict]:
    return [
        {
            "id": "NICE-DEVICE",
            "source_id": "NICE_NG136",
            "authority": "National Institute for Health and Care Excellence (NICE)",
            "title": "Hypertension in adults: diagnosis and management",
            "version": "test-v1",
            "locator": "Recommendation 1.1.3",
            "url": "https://example.org/nice",
            "kind": "guideline",
            "population": "Adults aged 18 years and older",
            "population_scope": {"included": ["adult"], "excluded": []},
            "text": "成人血压设备应经过验证、维护，并按制造商说明定期校准。",
            "topics": ["测量设备", "设备校准", "心律不齐", "人工听诊"],
        },
        {
            "id": "NICE-HOME",
            "source_id": "NICE_NG136",
            "authority": "National Institute for Health and Care Excellence (NICE)",
            "title": "Hypertension in adults: diagnosis and management",
            "version": "test-v1",
            "locator": "Recommendation 1.2.7",
            "url": "https://example.org/nice",
            "kind": "guideline",
            "population": "Adults aged 18 years and older",
            "population_scope": {"included": ["adult"], "excluded": []},
            "text": "家庭血压监测应按标准操作并记录多次读数。",
            "topics": ["家庭血压监测", "读数记录"],
        },
        {
            "id": "AHA-NIGHT",
            "source_id": "AHA_ACC_2025",
            "authority": "American Heart Association/American College of Cardiology (AHA/ACC)",
            "title": "2025 Guideline for High Blood Pressure in Adults",
            "version": "test-v1",
            "locator": "Table 3",
            "url": "https://example.org/aha",
            "kind": "guideline",
            "population": "Adults",
            "population_scope": {"included": ["adult"], "excluded": []},
            "text": "动态血压监测可以分别报告夜间和二十四小时平均读数。",
            "topics": ["动态血压夜间读数", "二十四小时平均"],
        },
    ]


class ImprovedClinicalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.corpus_path = self.root / "evidence.jsonl"
        with self.corpus_path.open("w", encoding="utf-8") as handle:
            for record in _corpus():
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.backend = StubBackend()
        self.system = System(self.backend, str(self.corpus_path), str(self.root / "index"))

    def tearDown(self):
        self.temp.cleanup()

    def test_topic_plus_authority_accepts_chinese_domain_query_but_keeps_gates(self):
        query = "NICE 心律不齐时是否采用人工听诊？"
        self.assertIsNone(self.system._scope_reason(query, ()))
        self.assertEqual(self.system._state(), None)
        self.assertEqual(self.system._scope_reason("NICE 是否筛查结核？", ()), "out_of_scope")
        self.assertEqual(self.system._scope_reason("NICE 给我开降压药", ()), "personalized")
        self.assertEqual(self.system._scope_reason("NICE 妊娠期测量阈值", ()), "out_of_scope")

    def test_planner_accepts_only_exact_question_spans_and_preserves_scope(self):
        query = "NICE 家庭血压监测要几天；第一天是否计入平均？"
        self.backend.bad_planner = True
        facets, fallback = self.system._plan(query)
        self.assertIsNone(fallback)
        self.assertEqual(len(facets), 2)
        self.assertTrue(any("NICE" in facet for facet in facets))
        self.assertTrue(any("第一天是否计入平均" in facet for facet in facets))
        self.assertFalse(any("治疗管理建议" in facet for facet in facets))

    def test_single_retrieval_keeps_supplied_evidence_in_first_and_final_context(self):
        result = self.system.respond(
            "成人血压测量设备如何校准？",
            supplied=["NICE-HOME"],
            mode="single",
            params={"initial_k": 1, "max_evidence": 2},
        )
        self.assertIn("NICE-HOME", result["first_evidence"])
        self.assertIn("NICE-HOME", result["evidence"])
        self.assertEqual(result["engine"], "improved")

    def test_independent_facets_and_named_sources_are_covered(self):
        self.system._request_local.value = {
            "query": "NICE 与 AHA/ACC 测量比较", "supplied_ids": [], "protected_ids": [],
            "retrieval_events": [], "evidence_budget_overflow": 0,
        }
        try:
            params = self.system._params({})
            results = self.system._retrieve(
                ["NICE 测量设备", "AHA/ACC 动态血压夜间读数"], 2, params
            )
            ids = [record["id"] for record in results]
            authorities = {
                self.system.records[self.system._id_to_index[record_id]]["authority"]
                for record_id in ids
            }
            self.assertTrue(any("(NICE)" in authority for authority in authorities))
            self.assertTrue(any("(AHA/ACC)" in authority for authority in authorities))
            self.assertEqual(len(self.system._request_local.value["retrieval_events"]), 1)
        finally:
            del self.system._request_local.value

    def test_merge_protects_supplied_and_first_pass_supported_evidence(self):
        state = {
            "supplied_ids": ["NICE-HOME"],
            "protected_ids": ["NICE-DEVICE"],
            "evidence_budget_overflow": 0,
        }
        self.system._request_local.value = state
        try:
            records = {record["id"]: record for record in self.system.records}
            first = [
                dict(records["NICE-HOME"], _score=-1.0),
                dict(records["NICE-DEVICE"], _score=0.0),
                dict(records["AHA-NIGHT"], _score=0.9),
            ]
            second = [dict(records["AHA-NIGHT"], _score=1.0)]
            merged = self.system._merge_results(first, second, 2)
            self.assertEqual([item["id"] for item in merged], ["NICE-HOME", "NICE-DEVICE"])
        finally:
            del self.system._request_local.value

    def test_partial_answers_are_explicit_opt_in_and_keep_gap_unanswered(self):
        query = "NICE 成人血压测量需要什么；第一天如何处理？"
        full = self.system.respond(query, mode="single")
        self.assertTrue(full["abstain"])
        partial = self.system.respond(query, mode="single", params={"allow_partial": True})
        self.assertFalse(partial["abstain"])
        self.assertTrue(partial["partial"])
        self.assertIn("尚缺依据、未作答", partial["answer"])
        self.assertTrue(partial["citations"])
        self.assertTrue(partial["unanswered_facets"])

    def test_cli_keeps_baseline_as_default_and_exposes_partial_policy(self):
        parser = _build_parser()
        defaults = parser.parse_args(["--query", "NICE adult blood pressure test"])
        self.assertEqual(defaults.engine, "clinical")
        self.assertFalse(defaults.allow_partial)
        selected = parser.parse_args([
            "--query", "NICE adult blood pressure test", "--engine", "improved", "--allow-partial"
        ])
        self.assertEqual(selected.engine, "improved")
        self.assertTrue(selected.allow_partial)


if __name__ == "__main__":
    unittest.main()
