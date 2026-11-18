"""Regression checks for adult-first population scope and domain routing."""
import unittest

from clinical import System, _derive_population_scope


NICE_POPULATION = (
    "Adults aged 18 years and older, including adults with type 2 diabetes for "
    "measurement and diagnosis; pregnancy is addressed in a separate NICE guideline."
)
AHA_POPULATION = (
    "Adults; the guideline is not a source for pediatric or pregnancy-specific "
    "diagnostic care. Use only the stated adult classification and measurement sections."
)
NICE_CONTEXT = {
    "id": "NICE-HBPM-01",
    "kind": "guideline",
    "title": "Hypertension in adults: diagnosis and management",
    "authority": "NICE",
    "topics": ["HBPM", "home blood pressure", "measurement"],
    "population": NICE_POPULATION,
    "population_scope": _derive_population_scope(NICE_POPULATION),
}


class PopulationScopeTests(unittest.TestCase):
    def test_adult_guidelines_are_not_reclassified_by_exceptions(self):
        for population in (NICE_POPULATION, AHA_POPULATION):
            with self.subTest(population=population[:25]):
                scope = _derive_population_scope(population)
                self.assertIn("adult", scope["included"])
                self.assertTrue(System._population_compatible(
                    "adult blood pressure measurement", population, scope
                ))
                self.assertFalse(System._population_compatible(
                    "pediatric blood pressure measurement", population, scope
                ))
        pregnancy_only = _derive_population_scope(
            "Pregnant adults requiring pregnancy-specific blood-pressure management."
        )
        self.assertEqual(pregnancy_only["included"], ["pregnancy"])
        self.assertFalse(System._population_compatible(
            "adult blood pressure measurement",
            "Pregnant adults requiring pregnancy-specific blood-pressure management.",
            pregnancy_only,
        ))

    def test_explicit_population_scope_takes_precedence(self):
        adult_scope = {"included": ["adult"], "excluded": ["pregnancy"]}
        child_scope = {"included": ["pediatric"], "excluded": []}
        self.assertTrue(System._population_compatible(
            "adult blood pressure measurement", "legacy note mentions children",
            adult_scope,
        ))
        self.assertFalse(System._population_compatible(
            "adult blood pressure measurement", "Adults aged 18 years",
            child_scope,
        ))

    def test_domain_aliases_and_evidence_context_followups(self):
        for query in (
            "HBPM protocol in adults",
            "ABPM results in mmHg",
            "NICE high blood pressure screening",
            "家庭血压监测第一天如何处理？",
        ):
            with self.subTest(query=query):
                self.assertIsNone(System._scope_reason(query))
        self.assertIsNone(System._scope_reason(
            "第一天的数据是否纳入平均？", [NICE_CONTEXT]
        ))
        self.assertEqual(
            System._scope_reason("What is the weather?", [NICE_CONTEXT]),
            "out_of_scope",
        )
        self.assertEqual(
            System._scope_reason("我的血压是160/100，我需要服药吗？"),
            "personalized",
        )


if __name__ == "__main__":
    unittest.main()
