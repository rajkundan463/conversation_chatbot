from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TopicDetector:

    def __init__(
        self,
        threshold=0.45,
        min_topic_size=5
    ):

        self.threshold = threshold
        self.min_topic_size = min_topic_size

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed(self, text):

        if not text or not isinstance(text, str):
            text = ""

        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    def similarity(
        self,
        emb1,
        emb2
    ):

        return cosine_similarity(
            [emb1],
            [emb2]
        )[0][0]

    def _save_topic(
        self,
        topics,
        topic_id,
        current_topic
    ):

        if not current_topic:
            return

        topics.append({
            "topic_id": topic_id,
            "conversation_id":
                current_topic[0].get(
                    "conversation_id"
                ),

            "start_message":
                current_topic[0].get(
                    "global_message_id"
                ),

            "end_message":
                current_topic[-1].get(
                    "global_message_id"
                ),

            "message_count":
                len(current_topic),

            "messages":
                current_topic
        })

    def detect_topics(
        self,
        messages
    ):

        if not messages:
            return []

        topics = []

        current_topic = []

        topic_id = 1

        previous_conversation = None

        for msg in messages:

            if (
                "text" not in msg
                or
                "global_message_id" not in msg
            ):
                continue

            current_conversation = msg.get(
                "conversation_id"
            )

            # -----------------------------------
            # HARD CONVERSATION BOUNDARY
            # -----------------------------------

            if (
                previous_conversation is not None
                and
                current_conversation
                != previous_conversation
            ):

                self._save_topic(
                    topics,
                    topic_id,
                    current_topic
                )

                topic_id += 1

                current_topic = []

            previous_conversation = (
                current_conversation
            )

            current_topic.append(msg)

            if len(current_topic) < self.min_topic_size:
                continue

            recent_text = " ".join(
                [
                    m.get("text", "")
                    for m in current_topic[-5:]
                ]
            )

            topic_text = " ".join(
                [
                    m.get("text", "")
                    for m in current_topic[:-1]
                ]
            )

            if not topic_text.strip():
                continue

            recent_emb = self.embed(
                recent_text
            )

            topic_emb = self.embed(
                topic_text
            )

            score = self.similarity(
                recent_emb,
                topic_emb
            )

            if score < self.threshold:

                self._save_topic(
                    topics,
                    topic_id,
                    current_topic[:-1]
                )

                topic_id += 1

                current_topic = [
                    current_topic[-1]
                ]

        self._save_topic(
            topics,
            topic_id,
            current_topic
        )

        return topics