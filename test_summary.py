import json

from src.parser import ConversationParser
from src.topic_detector import TopicDetector
from src.summarizer import ConversationSummarizer


print("\nLoading conversations...")

parser = ConversationParser(
    "data/conversations.csv"
)

messages = parser.parse_all()

print(
    f"Total Messages: {len(messages)}"
)

print("\nDetecting topics...")

detector = TopicDetector()

topics = detector.detect_topics(
    messages[:500]
)

print(
    f"Topics Found: {len(topics)}"
)

summarizer = ConversationSummarizer()

topic_summaries = []

print("\nGenerating summaries...\n")

for topic in topics:

    result = summarizer.build_topic_summary(
        topic
    )

    topic_summaries.append(
        result
    )

print("=" * 80)

for topic in topic_summaries[:5]:

    print(
        f"\nTOPIC {topic['topic_id']}"
    )

    print(
        f"Messages: {topic['start_message']} -> {topic['end_message']}"
    )

    print(
        f"Title: {topic['topic_title']}"
    )

    print(
        f"Summary:\n{topic['summary']}"
    )

    print(
        "\nSample Messages:"
    )

    for msg in topic["sample_messages"]:

        print(
            f"- {msg}"
        )

    print(
        "\n" + "=" * 80
    )

with open(
    "outputs/topics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        topic_summaries,
        f,
        indent=4,
        ensure_ascii=False
    )

print(
    "\nSaved topics to outputs/topics.json"
)