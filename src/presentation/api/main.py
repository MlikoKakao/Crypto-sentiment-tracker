from fastapi import FastAPI

app = FastAPI()


@app.get("/ping")
def pong():
    return {"ping": "pong!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
