from pathlib import Path
from urllib.parse import quote
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import assistant
import database
import llm
import speech

app = FastAPI(title="AI Assistant: code help and voice data analysis")
BASE = Path(__file__).parent


class QuestionRequest(BaseModel):
    dataset_id: str
    question: str


class ChatRequest(BaseModel):
    question: str


class CodeRequest(BaseModel):
    code: str


class SqlRequest(BaseModel):
    dataset_id: str
    sql: str


@app.exception_handler(database.DatasetNotFound)
def dataset_not_found(request, error):
    return JSONResponse(status_code=404, content={"detail": str(error)})


@app.exception_handler(database.BadSQL)
def bad_sql(request, error):
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.exception_handler(requests.RequestException)
def llm_failed(request, error):
    return JSONResponse(status_code=502, content={"detail": "The language model request failed: " + str(error)})


def analyze(dataset_id, question):
    info = database.get_info(dataset_id)
    schema = database.schema_text(info)

    sql = llm.make_sql(question, schema)
    try:
        result = database.run_sql(dataset_id, sql)
    except database.BadSQL as e:
        # one more try: give the error back to the llm so it can fix its query
        sql = llm.make_sql(question, schema, error=str(e), bad_sql=sql)
        try:
            result = database.run_sql(dataset_id, sql)
        except database.BadSQL as e2:
            raise database.BadSQL(str(e2) + " | query: " + sql)

    try:
        explanation = llm.explain(question, result)
    except requests.RequestException:
        explanation = ""

    result["question"] = question
    result["explanation"] = explanation
    return result


def read_audio(audio, language):
    data = audio.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The audio file is empty")
    try:
        return speech.transcribe(data, language or None)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not read the audio: " + str(e))


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.post("/datasets")
def upload_dataset(file: UploadFile = File(...)):
    content = file.file.read()
    try:
        dataset_id = database.save_dataset(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the file, is it a valid csv or xlsx?")
    return database.get_info(dataset_id)


@app.get("/datasets/{dataset_id}")
def dataset_info(dataset_id: str):
    return database.get_info(dataset_id)


@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...), language: str = Form("")):
    text, detected = read_audio(audio, language)
    return {"text": text, "language": detected}


@app.post("/generate-sql")
def generate_sql(body: QuestionRequest):
    schema = database.schema_text(database.get_info(body.dataset_id))
    return {"question": body.question, "sql": llm.make_sql(body.question, schema)}


@app.post("/execute-sql")
def execute_sql(body: SqlRequest):
    return database.run_sql(body.dataset_id, body.sql)


@app.post("/analyze-text")
def analyze_text(body: QuestionRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="The question is empty")
    return analyze(body.dataset_id, body.question.strip())


@app.post("/analyze-voice")
def analyze_voice(dataset_id: str = Form(...), audio: UploadFile = File(...), language: str = Form("")):
    text, detected = read_audio(audio, language)
    if not text:
        raise HTTPException(status_code=400, detail="No speech was detected in the audio")
    result = analyze(dataset_id, text)
    result["transcription"] = text
    result["language"] = detected
    return result


# ---------- code assistant ----------

@app.post("/chat")
def chat(body: ChatRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="The question is empty")

    result = assistant.start_chat(question)
    pieces = result["body"]
    first = next(pieces, "")   # the llm request starts here, so a failure is a normal http error

    def stream():
        yield first
        yield from pieces

    headers = {
        "X-Intent": result["intent"],
        "X-Sources": quote(",".join(result["sources"])),
        "X-Needs-Solution": str(result["needs_solution"]).lower(),
    }
    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8", headers=headers)


@app.get("/chat/history")
def chat_history():
    return assistant.load_memory()["messages"]


@app.post("/chat/clear")
def chat_clear():
    memory = assistant.load_memory()
    memory["messages"] = []
    memory["waiting"] = None
    assistant.save_memory(memory)
    return {"cleared": True}


@app.post("/run-code")
def run_code(body: CodeRequest):
    return assistant.run_code(body.code)
