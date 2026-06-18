import json


class CheckpointManager:

    def __init__(
        self,
        summarizer,
        checkpoint_size=100
    ):

        self.summarizer = summarizer
        self.checkpoint_size = checkpoint_size

    def create_checkpoints(
        self,
        messages
    ):

        checkpoints = []

        checkpoint_id = 1

        for i in range(
            0,
            len(messages),
            self.checkpoint_size
        ):

            chunk = messages[
                i:i+self.checkpoint_size
            ]

            if not chunk:
                continue

            summary = self.summarizer.generate_summary(
                chunk,
                top_k=10
            )

            checkpoint = {

                "checkpoint_id":
                checkpoint_id,

                "start_message":
                chunk[0]["global_message_id"],

                "end_message":
                chunk[-1]["global_message_id"],

                "message_count":
                len(chunk),

                "summary":
                summary

            }

            checkpoints.append(
                checkpoint
            )

            checkpoint_id += 1

        return checkpoints

    def save(
        self,
        checkpoints,
        output_path
    ):

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                checkpoints,
                f,
                indent=4,
                ensure_ascii=False
            )