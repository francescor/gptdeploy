
from fastapi import FastAPI, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict

from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# from main import main

from typing import Dict, Any
from enum import Enum
import json

app = FastAPI()


class CreateRequest(BaseModel):
    task_id: str
    user_token: str
    test_scenario: str
    executor_description: str

class CreateResponse(BaseModel):
    result: Dict[str, Any]
    success: bool
    message: Optional[str]

class PROGRESS_TYPE(Enum):
    KEEPALIVE = "keepalive"
    INIT = "init"
    START = "start"
    CONSOLE = "console"
    PROGRESS = "progress"
    DONE = "done"
    WARNING = "warning"
    ERROR = "error"
    COMPLETE = "complete"

class Request():
    def __init__(self, data):
        self.status = PROGRESS_TYPE.INIT.value
        self.subject = "Request has been queued"
        self.task_id = data.task_id
        self.user_token = data.user_token
        self.test_scenario = data.test_scenario
        self.executor_description = data.executor_description
    def __str__(self):
        return f"""
        {self.status}
        {self.subject}
        {self.task_id}
        {self.user_token}
        {self.test_scenario}
        {self.executor_description}
        """

def report_progress(progressType: PROGRESS_TYPE, request: Request, subject: str):
    print("- Progress")
    request.status = progressType.value
    request.subject = subject
    print(request)

def process_request(request: Request):
    report_progress(PROGRESS_TYPE.INIT, request, subject="Request picked by the queue")
    # result = main(
    #     executor_description=request.executor_description,
    #     test_scenario=request.test_scenario,
    #     request=request,
    #     PROGRESS_TYPE=PROGRESS_TYPE,
    #     report_progress_callback=report_progress
    # )

@app.post("/microchain/api/v1/generate", response_model=CreateResponse)
def create_endpoint(request: CreateRequest, background_tasks: BackgroundTasks):
    r = Request(request)
    background_tasks.add_task(process_request, r)
    return CreateResponse(result=r.__dict__, success=True, message=None)

@app.get("/ping")
def ping():
    return "pong"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a custom exception handler for RequestValidationError
@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hubble_server:app", host="0.0.0.0", port=8888, log_level="info", reload=True)