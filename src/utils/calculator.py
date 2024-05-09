from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import os


def load_agent_tasks(agent_name):
    file_path = os.path.join(os.getcwd(), "agent_configs", f"{agent_name}_task.txt")
    with open(file_path) as f:
        task_inputs = f.readlines()
        return task_inputs


def get_numbers_concurrent(agent_list, agent_factory, llm_request_responses):
    agent_tasks = []
    for agent_name, agent_num in agent_list:
        task_input = load_agent_tasks(agent_name=agent_name)[0]
        for i in range(agent_num):
            agent = agent_factory.activate_agent(
                agent_name = agent_name,
                task_input = task_input
            )
            agent.start()
            agent_tasks.append(agent)

    for agent in agent_tasks:
        agent.join()

    stats = {
        'agent_waiting_times': [],
        'agent_turnaround_times': [],
        'request_waiting_times': [],
        'request_turnaround_times': []
    }

    # Collect data

    for agent in agent_tasks:
        output = llm_request_responses.pop(agent.get_aid())
        agent_factory.deactivate_agent(agent.get_aid())
        agent.terminate()
        stats['agent_waiting_times'].append(output["agent_waiting_time"])
        stats['agent_turnaround_times'].append(output["agent_turnaround_time"])
        stats['request_waiting_times'].extend(output["request_waiting_times"])
        stats['request_turnaround_times'].extend(output["request_turnaround_times"])

    # Compute averages and percentiles
    def compute_metrics(data):
        return {
            'avg': np.mean(data),
            'p90': np.percentile(data, 90),
            'p99': np.percentile(data, 99)
        }

    metrics = {
        'agent_waiting_time': compute_metrics(stats['agent_waiting_times']),
        'agent_turnaround_time': compute_metrics(stats['agent_turnaround_times']),
        'request_waiting_time': compute_metrics(stats['request_waiting_times']),
        'request_turnaround_time': compute_metrics(stats['request_turnaround_times'])
    }

    return metrics


def get_numbers_sequential(agent_list, agent_factory, llm_request_responses):
    agent_tasks = []
    for agent_name, agent_num in agent_list:
        task_input = load_agent_tasks(agent_name=agent_name)[0]
        for i in range(agent_num):
            agent = agent_factory.activate_agent(
                agent_name = agent_name,
                task_input = task_input
            )
            agent.start()
            agent.join()
            agent_tasks.append(agent)

    stats = {
        'agent_waiting_times': [],
        'agent_turnaround_times': [],
        'request_waiting_times': [],
        'request_turnaround_times': []
    }

    # Collect data
    accumulated_time = 0
    for agent in agent_tasks:
        output = llm_request_responses.pop(agent.get_aid())
        agent_factory.deactivate_agent(agent.get_aid())
        agent.terminate()

        stats['agent_waiting_times'].append(output["agent_waiting_time"] + accumulated_time)

        stats['agent_turnaround_times'].append(output["agent_turnaround_time"] + accumulated_time)

        output["request_waiting_times"][0] += accumulated_time
        stats['request_waiting_times'].extend(
            output["request_waiting_times"]
        )

        output["request_turnaround_times"][0] += accumulated_time

        stats['request_turnaround_times'].extend(output["request_turnaround_times"])

        accumulated_time += (output['agent_turnaround_time'] - output["agent_waiting_time"])

    # Compute averages and percentiles
    def compute_metrics(data):
        return {
            'avg': np.mean(data),
            'p90': np.percentile(data, 90),
            'p99': np.percentile(data, 99)
        }

    metrics = {
        'agent_waiting_time': compute_metrics(stats['agent_waiting_times']),
        'agent_turnaround_time': compute_metrics(stats['agent_turnaround_times']),
        'request_waiting_time': compute_metrics(stats['request_waiting_times']),
        'request_turnaround_time': compute_metrics(stats['request_turnaround_times'])
    }

    return metrics


def calculate_improvement(sequential, concurrent):
    return (sequential - concurrent) / sequential


def comparison(concurrent_metrics, sequential_metrics):
    # Print analysis
    print("**** Improvement Analysis Starts ****")

    # Calculate and print improvements for averages and percentiles
    improvements = {}
    for key in concurrent_metrics.keys():
        concurrent_avg = concurrent_metrics[key]['avg']
        sequential_avg = sequential_metrics[key]['avg']
        improvements[key + '_avg_improv'] = calculate_improvement(sequential_avg, concurrent_avg) * 100

        concurrent_p90 = concurrent_metrics[key]['p90']
        sequential_p90 = sequential_metrics[key]['p90']
        improvements[key + '_p90_improv'] = calculate_improvement(sequential_p90, concurrent_p90) * 100

        concurrent_p99 = concurrent_metrics[key]['p99']
        sequential_p99 = sequential_metrics[key]['p99']
        improvements[key + '_p99_improv'] = calculate_improvement(sequential_p99, concurrent_p99) * 100

    # Print improvements
    for improv_key, improv_value in improvements.items():
        print(f"Improvement of {improv_key}: {improv_value:.2f}%")
