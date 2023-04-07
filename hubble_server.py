
from fastapi import FastAPI, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict

from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from typing import Dict, Any
from enum import Enum
import json
import random
import requests
import os
import re
import subprocess
import webbrowser
from pathlib import Path
import hashlib

import click

from src import gpt, jina_cloud
from src.jina_cloud import process_error_message, jina_auth_login
from src.prompt_tasks import general_guidelines, executor_file_task, chain_of_thought_creation, test_executor_file_task, \
    chain_of_thought_optimization, requirements_file_task, docker_file_task, not_allowed
from src.utils.io import recreate_folder, persist_file
from src.utils.string_tools import print_colored
from src.constants import FILE_AND_TAG_PAIRS

import hubble
from hubble.executor.helper import upload_file, archive_package, get_request_header
from jcloud.flow import CloudFlow

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







def push_executor(dir_path):
    dir_path = Path(dir_path)

    md5_hash = hashlib.md5()
    bytesio = archive_package(dir_path)
    content = bytesio.getvalue()
    md5_hash.update(content)
    md5_digest = md5_hash.hexdigest()

    form_data = {
        'public': 'True',
        'private': 'False',
        'verbose': 'True',
        'md5sum': md5_digest,
    }
    req_header = get_request_header()
    resp = upload_file(
        f'{os.getenv("JINA_HUBBLE_REGISTRY", "https://api.hubble.jina.ai")}/v2/rpc/executor.push',
        'filename',
        content,
        dict_data=form_data,
        headers=req_header,
        stream=False,
        method='post',
    )
    json_lines_str = resp.content.decode('utf-8')
    if 'exited on non-zero code' not in json_lines_str:
        return ''
    responses = []
    for json_line in json_lines_str.splitlines():
        if 'exit code:' in json_line:
            break
        d = json.loads(json_line)
        if 'payload' in d and type(d['payload']) == str:
            responses.append(d['payload'])
        elif type(d) == str:
            responses.append(d)
    return '\n'.join(responses)










def extract_content_from_result(plain_text, file_name):
    pattern = fr"^\*\*{file_name}\*\*\n```(?:\w+\n)?([\s\S]*?)```"
    match = re.search(pattern, plain_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    else:
        return ''

def write_config_yml(executor_name, dest_folder):
    config_content = f'''
jtype: {executor_name}
py_modules:
  - executor.py
metas:
  name: {executor_name}
    '''
    with open(os.path.join(dest_folder, 'config.yml'), 'w') as f:
        f.write(config_content)

def get_all_executor_files_with_content(folder_path):
    file_name_to_content = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                file_name_to_content[filename] = content

    return file_name_to_content

def files_to_string(file_name_to_content):
    all_executor_files_string = ''
    for file_name, tag in FILE_AND_TAG_PAIRS:
        if file_name in file_name_to_content:
            all_executor_files_string += f'**{file_name}**\n'
            all_executor_files_string += f'```{tag}\n'
            all_executor_files_string += file_name_to_content[file_name]
            all_executor_files_string += '\n```\n\n'
    return all_executor_files_string


def wrap_content_in_code_block(executor_content, file_name, tag):
    return f'**{file_name}**\n```{tag}\n{executor_content}\n```\n\n'


def create_executor(
        description,
        test,
        output_path,
        executor_name,
        package,
        is_chain_of_thought=False,
):
    EXECUTOR_FOLDER_v1 = get_executor_path(output_path, package, 1)
    recreate_folder(EXECUTOR_FOLDER_v1)
    recreate_folder('flow')

    print_colored('', '############# Executor #############', 'red')
    user_query = (
            general_guidelines()
            + executor_file_task(executor_name, description, test, package)
            + chain_of_thought_creation()
    )
    conversation = gpt.Conversation()
    executor_content_raw = conversation.query(user_query)
    if is_chain_of_thought:
        executor_content_raw = conversation.query(
            f"General rules: " + not_allowed() + chain_of_thought_optimization('python', 'executor.py'))
    executor_content = extract_content_from_result(executor_content_raw, 'executor.py')

    persist_file(executor_content, os.path.join(EXECUTOR_FOLDER_v1, 'executor.py'))

    print_colored('', '############# Test Executor #############', 'red')
    user_query = (
            general_guidelines()
            + wrap_content_in_code_block(executor_content, 'executor.py', 'python')
            + test_executor_file_task(executor_name, test)
    )
    conversation = gpt.Conversation()
    test_executor_content_raw = conversation.query(user_query)
    if is_chain_of_thought:
        test_executor_content_raw = conversation.query(
            f"General rules: " + not_allowed() +
            chain_of_thought_optimization('python', 'test_executor.py')
            + "Don't add any additional tests. "
        )
    test_executor_content = extract_content_from_result(test_executor_content_raw, 'test_executor.py')
    persist_file(test_executor_content, os.path.join(EXECUTOR_FOLDER_v1, 'test_executor.py'))

    print_colored('', '############# Requirements #############', 'red')
    user_query = (
            general_guidelines()
            + wrap_content_in_code_block(executor_content, 'executor.py', 'python')
            + wrap_content_in_code_block(test_executor_content, 'test_executor.py', 'python')
            + requirements_file_task()
    )
    conversation = gpt.Conversation()
    requirements_content_raw = conversation.query(user_query)
    if is_chain_of_thought:
        requirements_content_raw = conversation.query(
            chain_of_thought_optimization('', 'requirements.txt') + "Keep the same version of jina ")

    requirements_content = extract_content_from_result(requirements_content_raw, 'requirements.txt')
    persist_file(requirements_content, os.path.join(EXECUTOR_FOLDER_v1,'requirements.txt'))

    print_colored('', '############# Dockerfile #############', 'red')
    user_query = (
            general_guidelines()
            + wrap_content_in_code_block(executor_content, 'executor.py', 'python')
            + wrap_content_in_code_block(test_executor_content, 'test_executor.py', 'python')
            + wrap_content_in_code_block(requirements_content, 'requirements.txt', '')
            + docker_file_task()
    )
    conversation = gpt.Conversation()
    dockerfile_content_raw = conversation.query(user_query)
    if is_chain_of_thought:
        dockerfile_content_raw = conversation.query(
            f"General rules: " + not_allowed() + chain_of_thought_optimization('dockerfile', 'Dockerfile'))
    dockerfile_content = extract_content_from_result(dockerfile_content_raw, 'Dockerfile')
    persist_file(dockerfile_content, os.path.join(EXECUTOR_FOLDER_v1, 'Dockerfile'))

    write_config_yml(executor_name, EXECUTOR_FOLDER_v1)


def create_playground(executor_name, executor_path, host):
    print_colored('', '############# Playground #############', 'red')

    file_name_to_content = get_all_executor_files_with_content(executor_path)
    user_query = (
            general_guidelines()
            + wrap_content_in_code_block(file_name_to_content['executor.py'], 'executor.py', 'python')
            + wrap_content_in_code_block(file_name_to_content['test_executor.py'], 'test_executor.py', 'python')
            + f'''
Create a playground for the executor {executor_name} using streamlit. 
The executor is hosted on {host}. 
This is an example how you can connect to the executor assuming the document (d) is already defined:
from jina import Client, Document, DocumentArray
client = Client(host='{host}')
response = client.post('/', inputs=DocumentArray([d])) # always use '/'
print(response[0].text) # can also be blob in case of image/audio..., this should be visualized in the streamlit app
'''
    )
    conversation = gpt.Conversation()
    conversation.query(user_query)
    playground_content_raw = conversation.query(
        f"General rules: " + not_allowed() + chain_of_thought_optimization('python', 'app.py'))
    playground_content = extract_content_from_result(playground_content_raw, 'app.py')
    persist_file(playground_content, os.path.join(executor_path, 'app.py'))

def get_executor_path(output_path, package, version):
    package_path = '_'.join(package)
    return os.path.join(output_path, package_path, f'v{version}')

def debug_executor(output_path, package, description, test, request):
    MAX_DEBUGGING_ITERATIONS = 10
    error_before = ''
    for i in range(1, MAX_DEBUGGING_ITERATIONS):
        report_progress(PROGRESS_TYPE.PROGRESS, request, subject=f"Debugging attempt: {i}/{MAX_DEBUGGING_ITERATIONS}")
        previous_executor_path = get_executor_path(output_path, package, i)
        next_executor_path = get_executor_path(output_path, package, i + 1)
        log_hubble = push_executor(previous_executor_path)
        error = process_error_message(log_hubble)
        if error:
            recreate_folder(next_executor_path)
            file_name_to_content = get_all_executor_files_with_content(previous_executor_path)
            all_files_string = files_to_string(file_name_to_content)
            user_query = (
                    f"General rules: " + not_allowed()
                    + 'Here is the description of the task the executor must solve:\n'
                    + description
                    + '\n\nHere is the test scenario the executor must pass:\n'
                    + test
                    + 'Here are all the files I use:\n'
                    + all_files_string
                    + (('This is an error that is already fixed before:\n'
                        + error_before) if error_before else '')
                    + '\n\nNow, I get the following error:\n'
                    + error + '\n'
                    + 'Think quickly about possible reasons. '
                      'Then output the files that need change. '
                      "Don't output files that don't need change. "
                      "If you output a file, then write the complete file. "
                      "Use the exact same syntax to wrap the code:\n"
                      f"**...**\n"
                      f"```...\n"
                      f"...code...\n"
                      f"```\n\n"
            )
            conversation = gpt.Conversation()
            returned_files_raw = conversation.query(user_query)
            for file_name, tag in FILE_AND_TAG_PAIRS:
                updated_file = extract_content_from_result(returned_files_raw, file_name)
                if updated_file:
                    file_name_to_content[file_name] = updated_file

            for file_name, content in file_name_to_content.items():
                persist_file(content, os.path.join(next_executor_path, file_name))
            error_before = error

        else:
            break
        if i == MAX_DEBUGGING_ITERATIONS - 1:
            raise MaxDebugTimeReachedException('Could not debug the executor.')
    return get_executor_path(output_path, package, i)

class MaxDebugTimeReachedException(BaseException):
    pass


def generate_executor_name(description):
    conversation = gpt.Conversation()
    user_query = f'''
Generate a name for the executor matching the description:
"{description}"
The executor name must fulfill the following criteria:
- camel case
- start with a capital letter
- only consists of lower and upper case characters
- end with Executor.

The output is a the raw string wrapped into ``` and starting with **name.txt** like this:
**name.txt**
```
PDFParserExecutor
```
'''
    name_raw = conversation.query(user_query)
    name = extract_content_from_result(name_raw, 'name.txt')
    return name

def get_possible_packages(description, threads):
    print_colored('', '############# What package to use? #############', 'red')
    user_query = f'''
Here is the task description of the problme you need to solve:
"{description}"
First, write down all the subtasks you need to solve which require python packages.
For each subtask:
    Provide a list of 1 to 3 python packages you could use to solve the subtask. Prefer modern packages.
    For each package:
        Write down some non-obvious thoughts about the challenges you might face for the task and give multiple approaches on how you handle them.
        For example, there might be some packages you must not use because they do not obay the rules:
        {not_allowed()}
        Discuss the pros and cons for all of these packages.
Create a list of package subsets that you could use to solve the task.
The list is sorted in a way that the most promising subset of packages is at the top.
The maximum length of the list is 5.

The output must be a list of lists wrapped into ``` and starting with **packages.csv** like this:
**packages.csv**
```
package1,package2
package2,package3,...
...
```
    '''
    conversation = gpt.Conversation()
    packages_raw = conversation.query(user_query)
    packages_csv_string = extract_content_from_result(packages_raw, 'packages.csv')
    packages = [package.split(',') for package in packages_csv_string.split('\n')]
    packages = packages[:threads]
    return packages












def report_progress(progressType: PROGRESS_TYPE, request: Request, subject: str):
    try:
        print("- Progress")
        request.status = progressType.value
        request.subject = subject
        print(request)
        url = f'{os.getenv("JINA_HUBBLE_REGISTRY", "https://api.hubble.jina.ai")}/v2/rpc/executor.updateExecutorGenerationSession'
        data = {
            "taskId": request.task_id,
            "type": request.status,
            "subject": request.subject,
            "payload": {}
        }
        headers = {
            "Authorization": f"Bearer {request.user_token}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, data=data)
        print(f"""
        Response:
        {response.text}
        """)  
    except Exception as e:
        print(f"Report error: {e}")

# TODO: deal with hardcoded url in jina_cloud.py
# TODO: deal with auth issue when pusshing executor & flow

def process_request(request: Request):
    NUM_APPROACHES = 3
    OUTPUT_PATH = f"executor-{request.task_id}"
    description = request.executor_description
    test = request.test_scenario
    try:
        report_progress(PROGRESS_TYPE.INIT, request, subject="Request picked by the queue")
        client = hubble.Client(max_retries=None, jsonify=True, token=request.user_token)
        print(client.get_user_info())
        report_progress(PROGRESS_TYPE.START, request, subject="Process started")
        report_progress(PROGRESS_TYPE.PROGRESS, request, subject="Prep step started")
        generated_name = generate_executor_name(description)
        executor_name = f'{generated_name}{random.randint(0, 1000_000)}'
        packages_list = get_possible_packages(description, NUM_APPROACHES)
        recreate_folder(OUTPUT_PATH)
        report_progress(PROGRESS_TYPE.PROGRESS, request, subject="Prep step done")
        for packages in packages_list:
            try:
                report_progress(PROGRESS_TYPE.PROGRESS, request, subject=f"Creating executor: {executor_name}")
                create_executor(description, test, OUTPUT_PATH, executor_name, packages)
                report_progress(PROGRESS_TYPE.PROGRESS, request, subject=f"Executor created: {executor_name}")
                report_progress(PROGRESS_TYPE.PROGRESS, request, subject=f"Debugging executor: {executor_name}")
                executor_path = debug_executor(OUTPUT_PATH, packages, description, test, request)
                report_progress(PROGRESS_TYPE.PROGRESS, request, subject=f"Executor debugged: {executor_name}")
                report_progress(PROGRESS_TYPE.PROGRESS, request, subject="Deploying flow")
                host = jina_cloud.deploy_flow(executor_name, executor_path)
                report_progress(PROGRESS_TYPE.PROGRESS, request, subject=f"Flow deployed: {host}")
                #create_playground(executor_name, executor_path, host)
            except MaxDebugTimeReachedException:
                print('Could not debug the executor.')
                continue
            print(
                'Executor name:', executor_name, '\n',
                'Executor path:', executor_path, '\n',
                'Host:', host, '\n',
                'Playground:', f'streamlit run {os.path.join(executor_path, "app.py")}', '\n',
            )
            report_progress(PROGRESS_TYPE.COMPLETE, request, subject="Process done")
            break
    except Exception as e:
        print(f"Caught exception: {e}")
        report_progress(PROGRESS_TYPE.ERROR, request, subject="Something went wrong")

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