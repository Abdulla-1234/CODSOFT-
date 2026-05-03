"""
Task 1: Chatbot with Rule-Based Responses
CodSoft AI Internship
Author: Doodakula Mohammad Abdulla
"""

import re
from datetime import datetime


# ─── Knowledge Base ───────────────────────────────────────────────────────────

PATTERNS = [
    # Greetings
    (r"\b(hi|hello|hey|howdy|greetings|good\s*(morning|afternoon|evening))\b",
     ["Hello! 👋 How can I help you today?",
      "Hey there! What can I do for you?",
      "Hi! Nice to meet you. How may I assist?"]),

    # Name / Identity
    (r"\bwhat(?:'s| is) your name\b|\bwho are you\b",
     ["I'm RuBot — a rule-based AI assistant built for the CodSoft internship! 🤖",
      "My name is RuBot. I respond based on predefined patterns."]),

    # How are you
    (r"\bhow are you\b|\bhow do you do\b|\byou okay\b",
     ["I'm just a bot, but I'm functioning perfectly! 😄",
      "Doing great, thanks for asking! How about you?"]),

    # Thanks
    (r"\b(thanks|thank you|thx|ty)\b",
     ["You're welcome! 😊", "Happy to help!", "Anytime!"]),

    # Bye
    (r"\b(bye|goodbye|see you|cya|exit|quit)\b",
     ["Goodbye! Have a great day! 👋", "See you later! Take care!", "Bye! 😊"]),

    # Time
    (r"\bwhat(?:'s| is)(?: the)? time\b|\bcurrent time\b",
     [lambda: f"The current time is {datetime.now().strftime('%H:%M:%S')} 🕐"]),

    # Date
    (r"\bwhat(?:'s| is)(?: the)? date\b|\btoday(?:'s)? date\b",
     [lambda: f"Today is {datetime.now().strftime('%A, %B %d, %Y')} 📅"]),

    # Weather (canned)
    (r"\bweather\b",
     ["I can't check live weather yet, but you can try weather.com or just look outside! ☀️🌧️"]),

    # Jokes
    (r"\btell me a joke\b|\bfunny\b|\bjoke\b",
     ["Why do programmers prefer dark mode? Because light attracts bugs! 🐛😄",
      "Why did the AI go broke? It lost all its cache! 💸",
      "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads. 🍫"]),

    # Age
    (r"\bhow old are you\b|\bwhat(?:'s| is) your age\b",
     ["I was just born this internship cycle — so pretty young! 🍼"]),

    # Creator
    (r"\bwho (made|created|built|developed) you\b",
     ["I was built by Doodakula Mohammad Abdulla as part of the CodSoft AI Internship! 🎓"]),

    # Help
    (r"\bhelp\b|\bwhat can you do\b|\byour capabilities\b",
     [("I can help you with:\n"
       "  • Greetings & small talk\n"
       "  • Telling the time and date\n"
       "  • Jokes 😄\n"
       "  • Answering basic questions\n"
       "  • ... and more!\n"
       "Just type something and I'll respond!")]),

    # AI / Chatbot topic
    (r"\bwhat is (ai|artificial intelligence)\b",
     ["AI (Artificial Intelligence) is the simulation of human intelligence in machines — "
      "enabling them to learn, reason, and solve problems! 🧠"]),

    # Python
    (r"\bpython\b",
     ["Python is a fantastic programming language — great for AI, data science, and web dev! 🐍"]),

    # CodSoft
    (r"\bcodsoft\b",
     ["CodSoft is a vibrant internship platform that helps students develop skills in tech! 🚀"]),

    # Affirmations
    (r"\b(yes|yeah|yep|sure|absolutely|of course)\b",
     ["Great! Let me know what you need. 😊", "Awesome! How can I help?"]),

    # Negations
    (r"\b(no|nope|nah|not really)\b",
     ["That's alright! Is there anything else I can help with? 😊"]),

    # Insults / frustration (graceful handling)
    (r"\b(stupid|dumb|useless|hate you|shut up)\b",
     ["I'm sorry to hear that. I'm still learning! 😔 How can I do better?"]),

    # Love
    (r"\bi love you\b|\byou(?:'re| are) amazing\b",
     ["Aww, that's so sweet! 🥰 I'm just a bot, but I appreciate it!"]),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

import random

def get_response(user_input: str) -> str:
    """Match user input against patterns and return a response."""
    text = user_input.lower().strip()

    for pattern, responses in PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            choice = random.choice(responses)
            # Support callable responses (e.g., time/date lambdas)
            return choice() if callable(choice) else choice

    # Fallback
    fallbacks = [
        "Hmm, I'm not sure about that. Could you rephrase? 🤔",
        "I don't have an answer for that yet, but I'm learning!",
        "Interesting! Can you tell me more?",
        "I didn't quite get that. Try asking something else!",
    ]
    return random.choice(fallbacks)


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   🤖  RuBot — Rule-Based Chatbot  |  CodSoft AI")
    print("=" * 55)
    print("Type 'quit' or 'bye' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nRuBot: Goodbye! 👋")
            break

        if not user_input:
            continue

        response = get_response(user_input)
        print(f"RuBot: {response}\n")

        # Exit on farewell patterns
        if re.search(r"\b(bye|goodbye|exit|quit|cya)\b", user_input, re.IGNORECASE):
            break


if __name__ == "__main__":
    main()
