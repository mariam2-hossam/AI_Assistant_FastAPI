import json
import os
import subprocess
import sys
import tempfile
import knowledge
import llm

MEMORY_FILE = "memory.json"

NOT_FOUND = ("I couldn't find relevant knowledge for your request. Could you provide "
             "the correct solution or reference so I can learn it for future interactions?")
UNCLEAR = "I couldn't tell if you want an explanation or new code. Can you rephrase your request?"
SAVED = "Thanks, I saved your solution. Ask me again and I will use it."

languages = ["python", "java", "javascript", "c++", "c#"]
frameworks = ["flask", "django", "fastapi", "pytorch", "tensorflow", "streamlit"]


def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"messages": [], "language": "", "framework": "", "waiting": None}


def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def find_preferences(memory, text):
    text = text.lower()
    for lang in languages:
        if lang in text:
            memory["language"] = lang
    for fw in frameworks:
        if fw in text:
            memory["framework"] = fw


def build_messages(system, memory, question):
    if memory["language"]:
        system += " The user prefers " + memory["language"] + "."
    if memory["framework"]:
        system += " The user uses " + memory["framework"] + "."
    messages = [{"role": "system", "content": system}]
    messages += memory["messages"][-10:]
    messages.append({"role": "user", "content": question})
    return messages


def source_name(doc):
    return doc.metadata.get("task_id") or doc.metadata["source"]


def looks_like_solution(text):
    return "```" in text or text.count("\n") >= 3


def add_messages(memory, question, answer):
    memory["messages"].append({"role": "user", "content": question})
    memory["messages"].append({"role": "assistant", "content": answer})
    save_memory(memory)


def stream_and_save(stream, memory, question):
    # sends the pieces to the client and saves the full answer in the memory at the end
    parts = []
    try:
        for piece in stream:
            parts.append(piece)
            yield piece
    finally:
        if parts:
            add_messages(memory, question, "".join(parts))


def done(memory, question, intent, text, needs_solution=False):
    add_messages(memory, question, text)
    return {"intent": intent, "sources": [], "needs_solution": needs_solution, "body": iter([text])}


def start_chat(question):
    memory = load_memory()

    # we asked for a solution in the last answer, and this message is the reply
    waiting = memory["waiting"]
    memory["waiting"] = None
    if waiting and looks_like_solution(question):
        knowledge.add_solution(waiting, question)
        return done(memory, question, "learn", SAVED)

    find_preferences(memory, question)
    intent = llm.classify(question)

    if intent is None:
        return done(memory, question, "unclear", UNCLEAR)

    if intent == "EXPLAIN":
        # route 1: no rag
        system = "You are a programming tutor. Explain the code step by step."
        stream = llm.ask_stream(build_messages(system, memory, question))
        return {"intent": "explain", "sources": [], "needs_solution": False,
                "body": stream_and_save(stream, memory, question)}

    # route 2: rag, then relevance check, then the llm
    query = question
    old = [m["content"] for m in memory["messages"] if m["role"] == "user"]
    if len(question.split()) < 6 and old:
        query = old[-1] + " " + question

    docs = knowledge.search(query)
    context = "\n\n".join("Source: " + source_name(d) + "\n" + d.page_content for d in docs)
    if not docs or not llm.is_relevant(query, context):
        memory["waiting"] = question
        return done(memory, question, "generate", NOT_FOUND, needs_solution=True)

    system = ("You are a programming assistant. Write complete code in a markdown "
              "code block, with comments, error handling and best practices. "
              "Use the reference material if it helps.")
    full = question + "\n\nReference material:\n" + context
    stream = llm.ask_stream(build_messages(system, memory, full))
    return {"intent": "generate", "sources": sorted(set(source_name(d) for d in docs)),
            "needs_solution": False, "body": stream_and_save(stream, memory, question)}


def run_code(code):
    # temp folder + timeout, and the api key is not passed to the code
    env = {k: v for k, v in os.environ.items() if "KEY" not in k.upper()}
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            r = subprocess.run([sys.executable, path], cwd=folder, env=env, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=10)
            return {"stdout": r.stdout, "stderr": r.stderr}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Timeout: the code ran for more than 10 seconds"}
