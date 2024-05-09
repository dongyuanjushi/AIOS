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

    scheduler.start()

    agent_lists = [
        [
            "MathAgent",
            "Solve the problem that Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?"
        ],
        [
            "MathAgent",
            "Mark has 4 bags of marbles, each with 25 marbles. He gives 3 marbles to each of his 5 friends. How many marbles does he have left?"
        ],
        [
            "RecAgent",
            "I want to take a tour to New York during the spring break, recommend some restaurants around for me."
        ],
        [
            "NarrativeAgent",
            "I want to take a tour to New York during the spring break, recommend some restaurants around for me."
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
