from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
import re


class ConversationSummarizer:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        self.model = SentenceTransformer(
            model_name
        )

        self.stopwords = {
           "the","a","an","and","or","but",
           "is","are","was","were",
           "to","of","in","on","at",
           "for","with","from",
           "i","you","he","she","it",
           "we","they","me","my",
           "your","our","their",

           "really",
           "great",
           "good",
           "nice",
           "doing",
           "there",
           "thanks",
           "thank",
           "awesome",
           "love",
           "like",
           "yeah",
           "yes",
           "okay",
           "cool",
           "well",
           "hello",
           "hi"
        }

    def _get_texts(self, messages):

        texts = []

        for msg in messages:

            text = msg.get("text", "").strip()

            if text:
                texts.append(text)

        return texts

    def _extract_keywords(
        self,
        texts,
        top_n=5
    ):

        combined_text = " ".join(texts).lower()

        words = re.findall(
            r"\b[a-zA-Z]+\b",
            combined_text
        )

        filtered_words = [
            word
            for word in words
            if word not in self.stopwords
            and len(word) > 3
        ]

        common_words = Counter(
            filtered_words
        ).most_common(top_n)

        return [word for word, _ in common_words]

    def _representative_sentences(
        self,
        texts,
        top_k=5
    ):

        if len(texts) <= top_k:
            return texts

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        centroid = np.mean(
            embeddings,
            axis=0
        )

        similarities = cosine_similarity(
            embeddings,
            [centroid]
        ).flatten()

        ranked_idx = similarities.argsort()[::-1]

        selected = []
        selected_embeddings = []

        for idx in ranked_idx:

            candidate = texts[idx]
            candidate_emb = embeddings[idx]

            duplicate = False

            for emb in selected_embeddings:

                sim = cosine_similarity(
                    [candidate_emb],
                    [emb]
                )[0][0]

                if sim > 0.90:
                    duplicate = True
                    break

            if not duplicate:

                selected.append(candidate)
                selected_embeddings.append(
                    candidate_emb
                )

            if len(selected) >= top_k:
                break

        return selected

    def generate_summary(
        self,
        messages,
        top_k=5
    ):

        texts = self._get_texts(messages)

        if not texts:
            return ""

        representative = self._representative_sentences(
            texts,
            top_k
        )

        return " ".join(
            representative
        )

    def generate_topic_title(
        self,
        messages,
        top_n=4
    ):

        texts = self._get_texts(messages)

        if not texts:
            return "Unknown Topic"

        keywords = self._extract_keywords(
            texts,
            top_n
        )

        if not keywords:
            return "General Discussion"

        return " | ".join(
            word.capitalize()
            for word in keywords
        )

    def build_topic_summary(
        self,
        topic
    ):

        messages = topic["messages"]

        summary = self.generate_summary(
            messages
        )

        title = self.generate_topic_title(
            messages
        )

        return {

            "topic_id":
            topic["topic_id"],

            "start_message":
            topic["start_message"],

            "end_message":
            topic["end_message"],

            "message_count":
            len(messages),

            "topic_title":
            title,

            "summary":
            summary,

            "sample_messages": [

                msg["text"]

                for msg in messages[:3]

            ]
        }