# multiquery_ranker.py
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from llama_cpp import Llama
import os


class TinyLlamaQueryGenerator:
    def __init__(self, model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.llm = Llama(model_path=model_path, n_ctx=2048)

    def generate_queries(self, persona, task, num_queries=5):
        prompt = f"""You are an intelligent assistant designed to generate high-quality semantic search queries.
Your goal is to generate diverse and specific queries that would help retrieve the most relevant sections from a document collection.

Persona: {persona}
Task: {task}

Generate {num_queries} search queries (one per line) to retrieve information relevant to the task.
Examples:
1. Overview of the topic
2. Important entities or people involved
3. Steps or process explanation
4. Challenges and solutions
5. Tips or best practices

Now generate your own:"""

        output = self.llm(prompt, max_tokens=256)
        raw_text = output["choices"][0]["text"]
        print("🧠 Raw LLM Output:\n", raw_text)

        queries = []
        for line in raw_text.split("\n"):
            line = line.strip("\u2022-1234567890. ")
            if len(line) > 10:
                queries.append(f"Query: {line}")

        if not queries:
            print("⚠️ TinyLlama failed to generate valid queries. Falling back to default.")
            return [
                f"Query: overview of {task}",
                f"Query: key concepts in {task}",
                f"Query: steps involved in {task}",
                f"Query: problems and solutions in {task}",
                f"Query: best practices for {task}"
            ]

        return queries[:num_queries]


class MultiQueryRanker:
    def __init__(self,
                 model_name="models/bge-small-en-v1.5",
                 top_k=5,
                 max_chunks_per_doc=2,
                 use_tiny_llama=True,
                 llama_model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                 use_cross_encoder=True):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.max_chunks_per_doc = max_chunks_per_doc
        self.use_tiny_llama = use_tiny_llama
        self.use_cross_encoder = use_cross_encoder

        if self.use_tiny_llama:
            self.query_generator = TinyLlamaQueryGenerator(model_path=llama_model_path)

        if self.use_cross_encoder:
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def generate_queries(self, persona, task):
        if self.use_tiny_llama:
            return self.query_generator.generate_queries(persona, task)
        else:
            return [
                f"Query: overview of {task}",
                f"Query: key locations in {task}",
                f"Query: planning steps for {task}",
                f"Query: challenges in {task}",
                f"Query: must-see attractions in {task}"
            ]

    def rank(self, persona, task, chunks):
        queries = self.generate_queries(persona, task)
        if not queries:
            raise ValueError("No queries were generated.")

        query_embeddings = self.model.encode(queries)
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = self.model.encode(chunk_texts)

        similarity_scores = cosine_similarity(query_embeddings, chunk_embeddings)
        weights = [0.4] + [0.6 / (len(queries) - 1)] * (len(queries) - 1)
        weighted_scores = np.average(similarity_scores, axis=0, weights=weights)

        for i, chunk in enumerate(chunks):
            chunk["similarity"] = float(weighted_scores[i])

        ranked = sorted(chunks, key=lambda x: x["similarity"], reverse=True)

        # Limit by document
        doc_counter = {}
        filtered = []
        for chunk in ranked:
            doc = chunk["document"]
            if doc_counter.get(doc, 0) < self.max_chunks_per_doc:
                filtered.append(chunk)
                doc_counter[doc] = doc_counter.get(doc, 0) + 1
            if len(filtered) >= self.top_k:
                break

        # Optional: Cross-Encoder Re-Ranking
        if self.use_cross_encoder:
            pairs = [(task, chunk["text"]) for chunk in filtered]
            cross_scores = self.cross_encoder.predict(pairs)
            for i, chunk in enumerate(filtered):
                chunk["cross_score"] = float(cross_scores[i])
            filtered = sorted(filtered, key=lambda x: x["cross_score"], reverse=True)

        return filtered
