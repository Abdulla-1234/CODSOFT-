"""
Task 1: Chatbot with Rule-Based Responses
CodSoft AI Internship
Author: Doodakula Mohammad Abdulla
"""

import re
import random
from datetime import datetime


PATTERNS = [

    # ── Greetings ──────────────────────────────────────────────────────────
    (r"\b(hi|hello|hey|howdy|greetings|good\s*(morning|afternoon|evening|night))\b",
     ["Hello! 👋 How can I help you today?",
      "Hey there! What can I do for you?",
      "Hi! Nice to meet you. How may I assist?"]),

    # ── Identity ───────────────────────────────────────────────────────────
    (r"(what.?s your name|who are you|introduce yourself)",
     ["I'm RuBot — a rule-based AI assistant built for the CodSoft AI Internship! 🤖",
      "My name is RuBot. I respond based on predefined patterns and rules."]),

    # ── How are you ────────────────────────────────────────────────────────
    (r"how are you|how do you do|you okay|how.?re things",
     ["I'm just a bot, but I'm functioning perfectly! 😄",
      "Doing great, thanks for asking! How about you?",
      "All systems operational! 🟢 How can I help?"]),

    # ── Thanks ─────────────────────────────────────────────────────────────
    (r"\b(thanks|thank you|thx|ty|cheers)\b",
     ["You're welcome! 😊", "Happy to help!", "Anytime! 👍"]),

    # ── Bye ────────────────────────────────────────────────────────────────
    (r"\b(bye|goodbye|see you|cya|exit|quit|take care|farewell)\b",
     ["Goodbye! Have a great day! 👋",
      "See you later! Take care! 🙌"]),

    # ── Time ───────────────────────────────────────────────────────────────
    (r"(what.?s the time|current time|what time is it)",
     [lambda: f"The current time is {datetime.now().strftime('%H:%M:%S')} 🕐"]),

    # ── Date ───────────────────────────────────────────────────────────────
    (r"(what.?s the date|today.?s date|what day is it|current date)",
     [lambda: f"Today is {datetime.now().strftime('%A, %B %d, %Y')} 📅"]),

    # ── Weather ────────────────────────────────────────────────────────────
    (r"\bweather\b",
     ["I can't check live weather, but try weather.com or a quick Google search! ☀️🌧️"]),

    # ── Jokes ──────────────────────────────────────────────────────────────
    (r"(tell me a joke|say something funny|make me laugh|\bjoke\b)",
     ["Why do programmers prefer dark mode? Because light attracts bugs! 🐛😄",
      "Why did the AI go broke? It lost all its cache! 💸",
      "Why do Java developers wear glasses? Because they don't C#! 👓"]),

    # ── Help ───────────────────────────────────────────────────────────────
    (r"\bhelp\b|what can you do|your capabilities|what do you know",
     [("I can chat about:\n"
       "  💬 Small talk & greetings\n"
       "  🕐 Time & date\n"
       "  💻 Tech: Python, AI, ML, CI/CD, Git, APIs, Cloud, Security\n"
       "  🎓 Programming concepts (DSA, OOP, databases)\n"
       "  😄 Jokes\n"
       "Just ask me anything!")]),

    # ── Numbers only ───────────────────────────────────────────────────────
    (r"^\d[\d\s\+\-\*/\.]*$",
     ["That looks like a number or math expression! I'm a chatbot, not a calculator 😄",
      "Interesting number! I'm better at conversation — try typing a question!"]),

    # ═══════════════════════════════════════════════════════════════════════
    # TECH TOPICS
    # ═══════════════════════════════════════════════════════════════════════

    # ── CI/CD ──────────────────────────────────────────────────────────────
    (r"\b(ci.?cd|continuous integration|continuous (delivery|deployment)|pipeline|github actions|jenkins|gitlab ci|circleci|travis)\b",
     ["CI/CD = Continuous Integration / Continuous Delivery (Deployment) 🔄\n\n"
      "  CI — auto-builds & tests code every time you push a commit.\n"
      "  CD — auto-deploys tested code to staging or production.\n\n"
      "Popular tools: GitHub Actions, Jenkins, GitLab CI, CircleCI, Travis CI.\n"
      "Goal: catch bugs early, ship faster, with confidence! 🚀",

      "CI/CD automates the software delivery pipeline:\n"
      "  1. Push code → CI triggers automated tests\n"
      "  2. Tests pass → CD deploys the build automatically\n\n"
      "GitHub Actions uses YAML workflow files (.github/workflows/) to configure pipelines."]),

    # ── Git / Version Control ──────────────────────────────────────────────
    (r"\b(git\b|github|gitlab|version control|branching|merge|pull request|commit|repository|\brepo\b)\b",
     ["Git is a distributed version control system. 🗂️\n\n"
      "Key commands:\n"
      "  git init       — start a repo\n"
      "  git clone      — copy a repo\n"
      "  git add .      — stage changes\n"
      "  git commit -m  — save a snapshot\n"
      "  git push       — upload to remote\n"
      "  git pull       — download latest changes"]),

    # ── Python ─────────────────────────────────────────────────────────────
    (r"\bpython\b",
     ["Python is one of the most popular languages! 🐍\n"
      "Great for: AI/ML, Data Science, Web Dev, Automation.\n"
      "Key libraries: NumPy, Pandas, TensorFlow, PyTorch, scikit-learn, FastAPI."]),

    # ── AI / ML ────────────────────────────────────────────────────────────
    (r"\b(artificial intelligence|machine learning|deep learning|neural network|llm|large language model|transformer|nlp|computer vision)\b",
     ["AI simulates human intelligence in machines. 🧠\n\n"
      "  Machine Learning  — systems that learn from data\n"
      "  Deep Learning     — multi-layer neural networks\n"
      "  NLP               — understanding/generating language\n"
      "  Computer Vision   — understanding images & video"]),

    # ── APIs ───────────────────────────────────────────────────────────────
    (r"\b(api|restful|graphql|endpoint|http method|json api)\b",
     ["An API lets two software systems communicate. 🔌\n\n"
      "REST API methods:\n"
      "  GET    — retrieve data\n"
      "  POST   — create data\n"
      "  PUT    — update data\n"
      "  DELETE — remove data\n\n"
      "Data is usually exchanged in JSON format."]),

    # ── Cloud / DevOps ─────────────────────────────────────────────────────
    (r"\b(cloud computing|aws|azure|gcp|google cloud|ec2|lambda|kubernetes|docker|container|devops|microservice)\b",
     ["Cloud computing delivers computing over the internet. ☁️\n\n"
      "Top providers: AWS, Azure, GCP.\n"
      "Key services: virtual machines, storage, databases, serverless.\n\n"
      "Docker packages apps into containers 🐳\n"
      "Kubernetes (K8s) orchestrates containers at scale."]),

    # ── Databases ──────────────────────────────────────────────────────────
    (r"\b(database|sql\b|mysql|postgresql|mongodb|nosql|sqlite|orm)\b",
     ["Two main types of databases:\n\n"
      "  SQL (Relational): MySQL, PostgreSQL, SQLite — tables & relations\n"
      "  NoSQL: MongoDB, Redis, Cassandra — flexible schemas\n\n"
      "SQL example: SELECT * FROM users WHERE age > 18;"]),

    # ── Web Development ────────────────────────────────────────────────────
    (r"\b(web dev|html|css\b|javascript|react|vue|angular|node\.?js|django|flask|fastapi|frontend|backend|fullstack)\b",
     ["Web dev has two sides:\n\n"
      "  Frontend — what users see (HTML, CSS, JS, React/Vue)\n"
      "  Backend  — server logic (Python/Django/FastAPI, Node.js)\n"
      "  Fullstack — both combined\n\n"
      "Python frameworks: Django (full-featured), FastAPI (async), Flask (lightweight)."]),

    # ── DSA ────────────────────────────────────────────────────────────────
    (r"\b(data structure|algorithm|sorting|binary search|linked list|stack\b|queue\b|tree\b|graph\b|big.?o|time complexity)\b",
     ["Common Data Structures:\n\n"
      "  Array / List  — indexed collection\n"
      "  Stack         — LIFO (Last In, First Out)\n"
      "  Queue         — FIFO (First In, First Out)\n"
      "  Linked List   — pointer-linked nodes\n"
      "  Tree / Graph  — hierarchical / network data\n"
      "  Hash Map      — key-value pairs, O(1) lookup"]),

    # ── OOP ────────────────────────────────────────────────────────────────
    (r"\b(oop|object.oriented|class\b|inheritance|polymorphism|encapsulation|abstraction)\b",
     ["OOP (Object-Oriented Programming) organizes code into objects. 🧩\n\n"
      "4 pillars:\n"
      "  Encapsulation  — bundle data + methods together\n"
      "  Inheritance    — child class reuses parent class\n"
      "  Polymorphism   — same interface, different behaviour\n"
      "  Abstraction    — hide complexity, expose essentials"]),

    # ── Security ───────────────────────────────────────────────────────────
    (r"\b(security|cybersecurity|encryption|ssl|tls|https|firewall|vpn|oauth|jwt|authentication)\b",
     ["Cybersecurity protects systems from attacks. 🔒\n\n"
      "  Encryption  — scramble data (AES, RSA)\n"
      "  HTTPS/TLS   — secure data in transit\n"
      "  OAuth/JWT   — secure authentication standards\n"
      "  Firewall    — controls incoming/outgoing traffic"]),

    # ── Linux / OS ─────────────────────────────────────────────────────────
    (r"\b(linux|ubuntu|terminal|bash|command line|shell|kernel|chmod|grep)\b",
     ["Linux is the #1 OS for servers and AI dev. 🐧\n\n"
      "Essential commands:\n"
      "  ls, cd, mkdir, rm, cp, mv   — file management\n"
      "  grep, cat, tail, head       — text processing\n"
      "  chmod, sudo, ssh            — permissions & access"]),

    # ── CodSoft / Internship ───────────────────────────────────────────────
    (r"\b(codsoft|internship|intern|certificate|task|submission)\b",
     ["CodSoft is a vibrant internship platform for students in tech! 🚀\n"
      "Complete at least 3 AI tasks for a completion certificate.\n"
      "Submit your GitHub repo link and post a demo video on LinkedIn with #codsoft!"]),

    # ── Creator ────────────────────────────────────────────────────────────
    (r"who (made|created|built|developed) you|your (creator|developer|author)",
     ["Built by Doodakula Mohammad Abdulla for the CodSoft AI Internship 2026! 🎓"]),

    # ── Compliments ────────────────────────────────────────────────────────
    (r"\b(good bot|great|awesome|amazing|excellent|well done|nice|cool)\b",
     ["Thank you so much! 😊", "Glad I could help! 🎉", "You're too kind! 🥰"]),

    # ── Insults ────────────────────────────────────────────────────────────
    (r"\b(stupid|dumb|useless|hate you|shut up|idiot)\b",
     ["I'm sorry you feel that way! I'm still learning. 😔 How can I improve?"]),

    # ── Love ───────────────────────────────────────────────────────────────
    (r"\bi love you\b|you.?re amazing",
     ["Aww, that's sweet! 🥰 I'm just a bot, but I appreciate it!"]),
]


def get_response(user_input: str) -> str:
    text = user_input.strip()
    for pattern, responses in PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            choice = random.choice(responses)
            return choice() if callable(choice) else choice
    return random.choice([
        "Hmm, I'm not sure about that. Could you rephrase? 🤔",
        "Try asking about Python, AI, CI/CD, Git, APIs, or type 'help'!",
        "I didn't catch that — I'm better with tech topics. Type 'help' to see what I know!",
    ])


def main():
    print("=" * 57)
    print("   🤖  RuBot — Rule-Based Chatbot  |  CodSoft AI Task 1")
    print("=" * 57)
    print("Topics: tech, AI, Python, CI/CD, Git, APIs, cloud, jokes...")
    print("Type 'help' to see what I can do. Type 'bye' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nRuBot: Goodbye! 👋")
            break

        if not user_input:
            continue

        response = get_response(user_input)
        print(f"\nRuBot: {response}\n")

        if re.search(r"\b(bye|goodbye|exit|quit|cya|farewell)\b", user_input, re.IGNORECASE):
            break


if __name__ == "__main__":
    main()
