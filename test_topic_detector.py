# test_topic_detector.py

from src.parser import ConversationParser
from src.topic_detector import TopicDetector

parser = ConversationParser(
    "data/conversations.csv"
)

messages = parser.parse_all()

detector = TopicDetector()

topics = detector.detect_topics(
    messages[:500]
)

print("Topics:", len(topics))

for topic in topics[:5]:
    print(
        topic["topic_id"],
        topic["start_message"],
        topic["end_message"],
        len(topic["messages"])
    )