import os
from datasets import load_dataset
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from knowledge import add_documents, delete_source


# HumanEval: the prompt has the description, the solution is the code
delete_source("humaneval")
rows = load_dataset("openai/openai_humaneval", split="test")
docs = []
for row in rows:
    docs.append(Document(
        page_content=row["prompt"] + row["canonical_solution"],
        metadata={"source": "humaneval", "task_id": row["task_id"], "title": row["entry_point"]},
    ))
print("humaneval:", len(docs), "problems ->", add_documents(docs), "chunks")

# optional: any extra .md / .txt files you put in data/
if os.path.isdir("data"):
    for name in sorted(os.listdir("data")):
        if not name.endswith((".md", ".txt")):
            continue
        delete_source(name)
        files = TextLoader("data/" + name, encoding="utf-8").load()
        for d in files:
            d.metadata["source"] = name
            d.metadata["title"] = name.rsplit(".", 1)[0].replace("_", " ")
        print(name, "->", add_documents(files), "chunks")
