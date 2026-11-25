"""Small unit tests for the deterministic retrieval helpers."""
import unittest
from cybersage.retrieval import tokenize, reciprocal_rank_fusion


class RetrievalUtilityTests(unittest.TestCase):
    def test_tokenize_lowercases_and_splits(self):
        self.assertEqual(tokenize("Phishing EMAIL 101"), ["phishing", "email", "101"])

    def test_reciprocal_rank_fusion_merges_duplicate_chunks(self):
        semantic = [{"chunk_id": "a", "text": "alpha"}, {"chunk_id": "b", "text": "beta"}]
        keyword = [{"chunk_id": "b", "text": "beta"}, {"chunk_id": "c", "text": "gamma"}]
        result = reciprocal_rank_fusion([semantic, keyword])
        self.assertEqual(result[0]["chunk_id"], "b")
        self.assertEqual({item["chunk_id"] for item in result}, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
