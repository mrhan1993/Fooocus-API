from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

def start_app(args):
    uvicorn.run("fooocusapi.api:app", host=args.host, port=args.port, log_level=args.log_level)
