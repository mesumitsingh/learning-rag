import unittest

from rag import Document, RAGPipeline, VectorStore


class TestRAGPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            Document(id="doc1", text="Paris is the capital of France."),
            Document(id="doc2", text="Tokyo is the capital of Japan."),
            Document(id="doc3", text="Bananas are yellow fruits."),
        ]

    def test_vector_store_retrieves_most_relevant_document(self) -> None:
        store = VectorStore(self.documents)
        results = store.search("What is the capital of France?", k=1)

        self.assertEqual(results[0][0].id, "doc1")

    def test_pipeline_passes_context_to_llm(self) -> None:
        prompts = []

        def fake_llm(prompt: str) -> str:
            prompts.append(prompt)
            return "mock answer"

        pipeline = RAGPipeline(self.documents, llm_fn=fake_llm)
        answer = pipeline.answer("capital of japan", k=2)

        self.assertEqual(answer, "mock answer")
        self.assertEqual(len(prompts), 1)
        self.assertIn("[doc2] Tokyo is the capital of Japan.", prompts[0])
        self.assertIn("Question: capital of japan", prompts[0])


if __name__ == "__main__":
    unittest.main()
