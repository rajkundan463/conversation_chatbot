import json

from src.parser import ConversationParser
from src.persona_extractor import PersonaExtractor


parser = ConversationParser(
    "data/conversations.csv"
)

messages = parser.parse_all()

messages = messages[:2000]

extractor = PersonaExtractor()

persona = extractor.build_persona(
    messages
)

print(
    json.dumps(
        persona,
        indent=4
    )
)

with open(
    "outputs/persona.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        persona,
        f,
        indent=4,
        ensure_ascii=False
    )

print(
    "\nPersona saved."
)