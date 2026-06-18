import re
from collections import defaultdict, Counter


class PersonaExtractor:

    def __init__(self):

        self.hobby_patterns = [
            r"i like to (.+)",
            r"i enjoy (.+)",
            r"i love (.+)",
            r"i usually (.+)",
            r"i often (.+)"
        ]

        self.job_patterns = [
            r"i am a[n]?\s(.+)",
            r"i'm a[n]?\s(.+)",
            r"i work as a[n]?\s(.+)"
        ]

        self.study_patterns = [
            r"studying\s(.+)",
            r"study\s(.+)",
            r"majoring in\s(.+)"
        ]

    def clean_text(self, text):

        return text.strip().rstrip(".!?")

  
    # HABITS
   

    def extract_habits(self, messages):

        habits = defaultdict(list)

        for msg in messages:

            text = msg["text"]
            lower = text.lower()

            for pattern in self.hobby_patterns:

                match = re.search(pattern, lower)

                if match:

                    activities = match.group(1)

                    activities = re.split(
                        r",| and ",
                        activities
                    )

                    for activity in activities:

                        activity = activity.strip()

                        if len(activity) < 3:
                            continue

                        habits[
                            activity.title()
                        ].append(text)

        results = []

        for habit, evidence in habits.items():

            results.append({

                "habit": habit,

                "evidence":
                evidence[:3]

            })

        return sorted(
            results,
            key=lambda x: len(x["evidence"]),
            reverse=True
        )

    
    # PERSONAL FACTS
    

    def extract_personal_facts(self, messages):

        facts = []

        seen = set()

        for msg in messages:

            text = msg["text"]
            lower = text.lower()

            # Occupation

            for pattern in self.job_patterns:

                match = re.search(pattern, lower)

                if match:

                    role = self.clean_text(
                        match.group(1)
                    )

                    key = (
                        "occupation",
                        role
                    )

                    if key not in seen:

                        seen.add(key)

                        facts.append({

                            "category":
                            "occupation",

                            "fact":
                            role.title(),

                            "evidence":
                            [text]

                        })

            # Education

            for pattern in self.study_patterns:

                match = re.search(pattern, lower)

                if match:

                    subject = self.clean_text(
                        match.group(1)
                    )

                    key = (
                        "education",
                        subject
                    )

                    if key not in seen:

                        seen.add(key)

                        facts.append({

                            "category":
                            "education",

                            "fact":
                            subject.title(),

                            "evidence":
                            [text]

                        })

        return facts

    
    # PERSONALITY TRAITS
    

    def extract_traits(self, messages):

        scores = Counter()

        for msg in messages:

            text = msg["text"].lower()

            if "family" in text:
                scores["family_oriented"] += 1

            if "help people" in text:
                scores["helpful"] += 1

            if "learn" in text:
                scores["curious"] += 1

            if "adventure" in text:
                scores["adventurous"] += 1

            if "excited" in text:
                scores["enthusiastic"] += 1

            if "reading" in text:
                scores["intellectual"] += 1

            if "music" in text:
                scores["creative"] += 1

        traits = []

        total_messages = max(
            len(messages),
            1
        )

        for trait, score in scores.items():

            if score >= 2:

                traits.append({

                    "trait":
                    trait,

                    "confidence":
                    round(
                        score /
                        total_messages,
                        3
                    )

                })

        return sorted(
            traits,
            key=lambda x: x["confidence"],
            reverse=True
        )

    
    # COMMUNICATION STYLE
    

    def communication_style(self, messages):

        total_words = 0
        questions = 0
        exclamations = 0
        emojis = 0

        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )

        for msg in messages:

            text = msg["text"]

            total_words += len(
                text.split()
            )

            if "?" in text:
                questions += 1

            if "!" in text:
                exclamations += 1

            emojis += len(
                emoji_pattern.findall(text)
            )

        total_messages = max(
            len(messages),
            1
        )

        avg_words = (
            total_words /
            total_messages
        )

        tone = "neutral"

        if exclamations / total_messages > 0.20:
            tone = "enthusiastic"

        elif questions / total_messages > 0.30:
            tone = "curious"

        style = (
            "short"
            if avg_words < 12
            else "long"
        )

        return {

            "average_words_per_message":
            round(avg_words, 2),

            "question_ratio":
            round(
                questions /
                total_messages,
                2
            ),

            "exclamation_ratio":
            round(
                exclamations /
                total_messages,
                2
            ),

            "emoji_count":
            emojis,

            "tone":
            tone,

            "message_style":
            style
        }

    
    # BUILD PERSONA
    

    def build_persona(self, messages):

        return {

            "habits":
            self.extract_habits(
                messages
            ),

            "personal_facts":
            self.extract_personal_facts(
                messages
            ),

            "personality_traits":
            self.extract_traits(
                messages
            ),

            "communication_style":
            self.communication_style(
                messages
            )
        }