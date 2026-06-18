from src.parser import ConversationParser
from src.summarizer import ConversationSummarizer
from src.checkpoint_manager import CheckpointManager


print("Loading conversations...")

parser = ConversationParser(
    "data/conversations.csv"
)

messages = parser.parse_all()

print(
    f"Messages Loaded: {len(messages)}"
)

summarizer = ConversationSummarizer()

checkpoint_manager = CheckpointManager(
    summarizer=summarizer,
    checkpoint_size=100
)

print(
    "Generating checkpoints..."
)

checkpoints = checkpoint_manager.create_checkpoints(
    messages[:1000]
)

print(
    f"Checkpoints Created: {len(checkpoints)}"
)

for cp in checkpoints[:3]:

    print("\n" + "="*80)

    print(
        f"Checkpoint: {cp['checkpoint_id']}"
    )

    print(
        f"Messages: {cp['start_message']} -> {cp['end_message']}"
    )

    print(
        f"Count: {cp['message_count']}"
    )

    print(
        f"Summary:\n{cp['summary']}"
    )

checkpoint_manager.save(
    checkpoints,
    "outputs/checkpoints.json"
)

print(
    "\nSaved to outputs/checkpoints.json"
)
