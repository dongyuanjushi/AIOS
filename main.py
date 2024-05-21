import os
import sys
import json

from src.scheduler.fifo_scheduler import FIFOScheduler

from src.scheduler.rr_scheduler import RRScheduler

from src.utils.utils import (
    parse_global_args,
)

from openagi.src.agents.agent_factory import AgentFactory

# from openagi.src.agents.agent_process import AgentProcessFactory

import warnings

from src.llm_kernel import llms

# from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed

import multiprocessing

import psutil

import threading

import queue

from src.utils.utils import delete_directories
from dotenv import find_dotenv, load_dotenv

def clean_cache(root_directory):
    targets = {'.ipynb_checkpoints', '__pycache__', ".pytest_cache", "context_restoration"}
    delete_directories(root_directory, targets)

def main():
    warnings.filterwarnings("ignore")
    parser = parse_global_args()
    args = parser.parse_args()

    llm_name = args.llm_name
    max_gpu_memory = args.max_gpu_memory
    eval_device = args.eval_device
    max_new_tokens = args.max_new_tokens
    scheduler_log_mode = args.scheduler_log_mode
    agent_log_mode = args.agent_log_mode
    llm_kernel_log_mode = args.llm_kernel_log_mode
    load_dotenv()

    agent_process_queue = multiprocessing.Queue()
    # agent_process_queue = queue.Queue()

    llm_request_responses = multiprocessing.Manager().dict()
    # llm_request_responses = dict()

    llm_request_responses = multiprocessing.Manager().dict()

    llm = llms.LLMKernel(
        llm_name = llm_name,
        max_gpu_memory = max_gpu_memory,
        eval_device = eval_device,
        max_new_tokens = max_new_tokens,
        log_mode = llm_kernel_log_mode
    )

    scheduler = FIFOScheduler(
        llm = llm,
        agent_process_queue = agent_process_queue,
        llm_request_responses = llm_request_responses,
        log_mode = scheduler_log_mode
    )

    agent_factory = AgentFactory(
        llm = llm,
        agent_process_queue = agent_process_queue,
        llm_request_responses = llm_request_responses,
        agent_log_mode = agent_log_mode
    )

    # print(scheduler.cpu_affinity())

    scheduler.start()

    agent_lists = [
        [
            "TravelAgent",
            "I want to take a trip to Paris, France from July 4th to July 10th 2024 and I am traveling from New York City. Help me plan this trip."
        ],
        [
            "MathAgent",
            "Convert 15000 MXN to Canadian Dollars and find out how much it would be in USD if 1 CAD equals 0.79 USD."
        ],
        [
            "AcademicAgent",
            "Summarize recent advancements in quantum computing from the past five years."
        ],
        [
            "RecAgent",
            "Recommend two movies with groundbreaking visual effects released in the last fifteen years ranked between 1 and 20 with ratings above 8.0."
        ],
        [
            "CreationAgent",
            "Create an image of a lush jungle with an ancient temple, evoking a sense of mystery and adventure."
        ]
    ]

    agent_tasks = []

    for agent_name, task_input in agent_lists:
        agent = agent_factory.activate_agent(
            agent_name = agent_name,
            task_input = task_input
        )
        agent_tasks.append(agent)

        agent.start()

    for agent in agent_tasks:
        agent.join()

    for agent in agent_tasks:
        print(llm_request_responses.get(agent.get_aid()))

    scheduler.terminate()

    clean_cache(root_directory="./")

if __name__ == "__main__":
    main()
