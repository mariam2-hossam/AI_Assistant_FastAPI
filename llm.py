import json
import re
import requests
from config import API_KEY, MODEL, URL


def ask(messages):
    r = requests.post(URL, headers={"Authorization": "Bearer " + str(API_KEY)},
                      json={"model": MODEL, "messages": messages, "temperature": 0.8}, timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def guess_language(text):
    if re.search(r"[\u0600-\u06FF]", text):
        return "Arabic"
    return "English"


def clean_sql(text):
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    # if the model wrote a sentence before the query, start from the first SELECT / WITH
    start = re.search(r"\b(select|with)\b", text, re.IGNORECASE)
    if start:
        text = text[start.start():]
    return text.strip()


def make_sql(question, schema, error=None, bad_sql=None):
    system = ("You write SQLite queries. The question can be in Arabic or English. "
              "Use only the table and columns from the schema and copy their names exactly. "
              "Wrap every table name and column name in double quotes and write text values in single quotes. "
              "If the question mentions a value (a category or a city for example), match it to "
              "the values listed in the schema, even when the question is in another language. "
              "Return one SELECT query only, with no explanation and no markdown.")
    user = "Schema:\n" + schema + "\n\nQuestion: " + question
    if error:
        user += "\n\nYour previous query failed.\nQuery: " + str(bad_sql) + "\nError: " + error
        user += "\nWrite a corrected query."
    answer = ask([{"role": "system", "content": system}, {"role": "user", "content": user}])
    return clean_sql(answer)


def explain(question, result):
    language = guess_language(question)
    system = ("You are a data analyst. Explain the query result in 2 to 4 short sentences "
              "for someone who does not know SQL. Mention the important numbers. "
              "Answer in " + language + ".")
    rows = json.dumps(result["rows"][:20], ensure_ascii=False, default=str)
    user = ("Question: " + question + "\nSQL: " + result["sql"] + "\nColumns: "
            + ", ".join(result["columns"]) + "\nRows (" + str(result["row_count"]) + " in total): " + rows)
    return ask([{"role": "system", "content": system}, {"role": "user", "content": user}]).strip()


def ask_stream(messages):
    r = requests.post(URL, headers={"Authorization": "Bearer " + str(API_KEY)},
                      json={"model": MODEL, "messages": messages, "stream": True},
                      stream=True, timeout=90)
    r.raise_for_status()
    for line in r.iter_lines():
        line = line.decode("utf-8") if line else ""
        if not line.startswith("data:") or "[DONE]" in line:
            continue
        try:
            piece = json.loads(line[5:])["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, ValueError):
            continue
        if piece:
            yield piece


def classify(question):
    answer = ask([
        {"role": "system", "content": "Is the user asking to EXPLAIN existing code or to GENERATE new code? "
                                      "Answer with one word: EXPLAIN or GENERATE."},
        {"role": "user", "content": question},
    ]).upper()
    if "EXPLAIN" in answer and "GENERATE" not in answer:
        return "EXPLAIN"
    if "GENERATE" in answer and "EXPLAIN" not in answer:
        return "GENERATE"
    return None


def is_relevant(question, context):
    answer = ask([
        {"role": "system", "content": "Is the context useful for the request? Be strict. "
                                      "Answer only RELEVANT or NOT_RELEVANT."},
        {"role": "user", "content": "Request:\n" + question + "\n\nContext:\n" + context},
    ]).strip().upper()
    return answer.startswith("RELEVANT")
