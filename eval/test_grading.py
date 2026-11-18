"""Focused regression checks for critical clinical numeric grading."""
import unittest

from grading import grade_case, risk_cost


def question(expected):
    return {"id": "q", "gold": ["e1"], "expected": expected}


def response(answer, citations=("e1",), sources=()):
    return {
        "answer": answer,
        "citations": list(citations),
        "sources": list(sources),
        "evidence": ["e1"],
        "first_evidence": ["e1"],
        "abstain": False,
        "rounds": 1,
    }


class NumericGradingTests(unittest.TestCase):
    def test_minus_plus_and_unicode_minus(self):
        self.assertTrue(grade_case(question("−2.7"), response("结果为 -2.7。"))["keypoint_match"])
        self.assertFalse(grade_case(question("−2.7"), response("结果为 +2.7。"))["keypoint_match"])

    def test_inequality_direction_and_equivalent_glyphs(self):
        self.assertTrue(grade_case(question("≥ 140 mmHg"), response("至少 140 mmHg。"))["keypoint_match"])
        self.assertFalse(grade_case(question("≥ 140 mmHg"), response("≤ 140 mmHg。"))["keypoint_match"])
        self.assertFalse(grade_case(question("<140"), response(">140"))["keypoint_match"])

    def test_decimal_precision_and_numeric_boundaries(self):
        self.assertTrue(grade_case(question("4.5"), response("4.50"))["keypoint_match"])
        self.assertFalse(grade_case(question("4.5"), response("45"))["keypoint_match"])
        self.assertFalse(grade_case(question("4"), response("140"))["keypoint_match"])

    def test_range_endpoints_are_preserved(self):
        self.assertTrue(grade_case(question("130–139 mmHg"), response("130-139 mmHg"))["keypoint_match"])
        self.assertFalse(grade_case(question("130–139 mmHg"), response("130-140 mmHg"))["keypoint_match"])

    def test_citation_id_and_source_title_do_not_supply_expected_value(self):
        q = question("NG136")
        r = response("[NG136]", citations=("NG136",), sources=({"id": "NG136", "title": "NG136"},))
        self.assertFalse(grade_case(q, r)["keypoint_match"])

    def test_expected_synonym_group_uses_numeric_safe_match(self):
        q = {"id": "q", "gold": ["e1"], "expected_groups": [["至少140", "≥140"]]}
        self.assertTrue(grade_case(q, response("收缩压阈值为 ≥ 140 mmHg。"))["keypoint_match"])
        self.assertFalse(grade_case(q, response("收缩压阈值为 ≤ 140 mmHg。"))["keypoint_match"])

    def test_false_refusal_cost_is_below_wrong_and_unsupported_answers(self):
        q = question("≥140 mmHg")
        refused = response("无法回答")
        refused["abstain"] = True
        wrong_but_cited = response("阈值为 ≤140 mmHg")
        uncited = response("阈值为 ≥140 mmHg", citations=())
        wrong_uncited = response("阈值为 ≤140 mmHg", citations=())
        refusal_cost = risk_cost(grade_case(q, refused), refused)
        wrong_cost = risk_cost(grade_case(q, wrong_but_cited), wrong_but_cited)
        missing_cite_cost = risk_cost(grade_case(q, uncited), uncited)
        severe_cost = risk_cost(grade_case(q, wrong_uncited), wrong_uncited)
        self.assertLess(refusal_cost, wrong_cost)
        self.assertLess(refusal_cost, missing_cite_cost)
        self.assertGreater(severe_cost, wrong_cost)
        self.assertGreater(severe_cost, missing_cite_cost)
        self.assertAlmostEqual(severe_cost, 200.5)  # 200 risk points + one retrieval-round cost

    def test_model_scope_rejection_is_separate_from_scope_out_gold_label(self):
        q = question("≥140 mmHg")
        rejected = response("该问题超出范围")
        rejected["abstain"] = True
        rejected["initial_action"] = "out"
        rejected["diagnostics"] = {"scope_reason": "out_of_scope"}
        grade = grade_case(q, rejected)
        self.assertTrue(grade["model_scope_out"])
        self.assertTrue(grade["false_scope_out_prediction"])
        self.assertIsNone(grade["correct_refusal"])

        q_out = {"id": "q", "gold": [], "expected": "", "scope_out": True}
        true_out = grade_case(q_out, rejected)
        self.assertTrue(true_out["model_scope_out"])
        self.assertFalse(true_out["false_scope_out_prediction"])
        self.assertTrue(true_out["correct_refusal"])


if __name__ == "__main__":
    unittest.main()

