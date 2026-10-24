# AI Assistant (FastAPI)

One FastAPI app with two parts:

1. Code assistant: explains code or generates code (RAG over HumanEval in Qdrant),
   learns from the solutions you send, remembers the conversation, and can run python code.
2. Data analysis by voice: Voice -> Faster-Whisper -> text -> LLM -> SQL -> SQLite3 -> result + explanation.
   Works with Arabic and English.

## Run

    pip install -r requirements.txt
    copy .env.example to .env and add your OpenRouter key
    python ingest.py          (once, loads HumanEval into Qdrant, needs internet)
    python -m uvicorn main:app

Open http://127.0.0.1:8000 (page with both parts) or http://127.0.0.1:8000/docs (API docs).
The first voice request downloads the Whisper model (WHISPER_MODEL in .env: tiny, base, small, medium).

Qdrant saves the vectors in the qdrant_db folder, and only one program can open it at a time:
stop the server before running ingest.py. To use a Qdrant server instead, set QDRANT_URL in .env.

## Endpoints: code assistant

| Method | Path | What it does |
|---|---|---|
| POST | /chat | question -> classify -> explain (no RAG) or generate (Qdrant + relevance check). The answer is streamed. Headers: X-Intent, X-Sources, X-Needs-Solution |
| GET | /chat/history | the saved conversation |
| POST | /chat/clear | clears the conversation |
| POST | /run-code | runs python code (10 second limit) and returns stdout / stderr |

If nothing relevant is found the answer asks for a solution (X-Needs-Solution = true).
Send the solution as your next message (a code block or 4+ lines) and it is stored in Qdrant.

## Endpoints: data analysis

| Method | Path | What it does |
|---|---|---|
| POST | /datasets | upload a .csv or .xlsx file, it is loaded into SQLite, returns dataset_id and columns |
| GET | /datasets/{dataset_id} | table info and sample rows |
| POST | /transcribe | audio file -> text (Faster-Whisper) |
| POST | /generate-sql | question text -> SQL query (LLM), not executed |
| POST | /execute-sql | validates and runs a SELECT query on SQLite |
| POST | /analyze-text | question text -> SQL -> result + explanation |
| POST | /analyze-voice | audio -> text -> SQL -> result + explanation |

## Where the data goes

- Qdrant (vectors): only the programming knowledge for the code assistant (HumanEval + learned solutions).
- SQLite (tables): the dataset you upload for the data analysis. No vectors here, the LLM searches it with SQL.

## SQL safety

- Only one SELECT (or WITH ... SELECT) is accepted.
- The database is opened read only, and an authorizer blocks everything except reading.
- Queries stop after 5 seconds and return at most 500 rows.
- If the LLM writes a wrong query, the error is sent back to it once to fix the query.

## Important

/run-code runs code on the computer that runs the server, with only a temp folder and a time limit
(it is not a real sandbox). Run the server on your own computer (the default 127.0.0.1) and do not
open it to other people.

## Files

- main.py: all the endpoints
- assistant.py: code assistant logic (memory, routing, RAG flow, code runner)
- llm.py: all the LLM calls (classifier, relevance check, answers, SQL generation, explanation)
- knowledge.py: Qdrant (chunking, embeddings, search, learning from solutions)
- ingest.py: loads HumanEval into Qdrant
- database.py: SQLite (load the file, schema, SQL check and execution)
- speech.py: Faster-Whisper
- static/index.html: page with the two parts
- sample_data/: sales.csv and المبيعات.csv (same data, English and Arabic columns)
