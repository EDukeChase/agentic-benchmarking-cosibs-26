import hashlib
import tempfile
import unittest
from pathlib import Path

from src.settings import prompts
from src.settings.study_prompt import compose, load_study_prompt

PROMPT = "Predict things.\nData is in /home/llmrel/task/data.\n"


class StudyPromptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "prompt.md"
        self.path.write_text(PROMPT, encoding="utf-8")

    def test_hash_is_of_the_stored_bytes(self):
        prompt = load_study_prompt(self.path)
        self.assertEqual(prompt.sha256, hashlib.sha256(PROMPT.encode()).hexdigest())
        self.assertEqual(prompt.text, PROMPT)

    def test_expected_hash_is_enforced(self):
        digest = hashlib.sha256(PROMPT.encode()).hexdigest()
        load_study_prompt(self.path, digest)
        with self.assertRaises(ValueError):
            load_study_prompt(self.path, "0" * 64)

    def test_unrendered_template_is_refused(self):
        self.path.write_text("Data in {data_dir}", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_study_prompt(self.path)

    def test_compose_keeps_the_study_prompt_word_for_word(self):
        text = compose(load_study_prompt(self.path), "Stage text.")
        self.assertIn(PROMPT.rstrip(), text)
        self.assertTrue(text.rstrip().endswith("Stage text."))


class StagePromptTests(unittest.TestCase):
    def test_placeholders_are_all_supplied(self):
        prompts.LITERATURE_STAGE_INSTRUCTIONS.format(num_models=3)
        prompts.PROGRAMMING_STAGE_INSTRUCTIONS.format(name_mapping="- x -> y")
        prompts.BENCHMARKING_STAGE_INSTRUCTIONS.format(methods_dir="/m", out_dir="/o", data_dir="/d")

    def test_no_old_dataset_names_or_forbidden_words(self):
        for name in dir(prompts):
            if name.endswith("_INSTRUCTIONS"):
                text = getattr(prompts, name).lower()
                self.assertNotIn("ehrshot", text, name)
                self.assertNotIn("arm ", text, name)


if __name__ == "__main__":
    unittest.main()
