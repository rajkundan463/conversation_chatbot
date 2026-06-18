import pandas as pd
import re


class ConversationParser:

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load_data(self):

        df = pd.read_csv(
            self.csv_path,
            header=None
        )

        df.columns = ["conversation"]

        return df

    def parse_conversation(
        self,
        conversation_text,
        conversation_id
    ):

        messages = []

        pattern = r"(User \d+):\s*(.*)"

        lines = conversation_text.split("\n")

        message_id = 1

        for line in lines:

            match = re.match(pattern, line)

            if match:

                speaker = match.group(1)
                text = match.group(2).strip()

                messages.append({
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "speaker": speaker,
                    "text": text
                })

                message_id += 1

        return messages

    def parse_all(self):

        df = self.load_data()

        all_messages = []

        global_message_id = 1

        for idx, row in df.iterrows():

            conversation = str(row["conversation"])

            parsed = self.parse_conversation(
                conversation,
                idx
            )

            for msg in parsed:

                msg["global_message_id"] = global_message_id

                all_messages.append(msg)

                global_message_id += 1

        return all_messages