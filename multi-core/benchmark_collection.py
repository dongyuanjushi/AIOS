from openai import OpenAI

import google.generativeai as genai

from dotenv import load_dotenv

import json

import requests

import os

load_dotenv()

benchmarks = [
    # {"name": "MMLU", "url": "https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu"},
    # {"name": "GPQA", "url": "https://klu.ai/glossary/gpqa-eval"},
    # {"name": "MATH", "url": "https://paperswithcode.com/sota/math-word-problem-solving-on-math"},
    # {"name": "HumanEval", "url": "https://paperswithcode.com/sota/code-generation-on-humaneval"},
    # {"name": "GSM8K", "url": "https://paperswithcode.com/sota/arithmetic-reasoning-on-gsm8k"},
    # {"name": "Chatbot-arena", "url": "https://lmarena.ai/"}, # TODO bug
    # {"name": "LongBench", "url": "https://longbench2.github.io/#leaderboard"},
    # {"name": "AlpacaEval", "url": "https://tatsu-lab.github.io/alpaca_eval/"},
    # {"name": "CommonsenseQA", "url": "https://paperswithcode.com/sota/common-sense-reasoning-on-commonsenseqa"},
    # {"name": "BigCodeBench", "url": "https://bigcode-bench.github.io/"},
    # {"name": "EvalPlus", "url": "https://evalplus.github.io/leaderboard.html"}
    {"name": "DS1000", "url": "https://ds1000-code-gen.github.io/model_DS1000.html"}
]

with open("llm_benchmarks.json", "r") as f:
    final_results = json.load(f)

from tqdm import tqdm

# def fetch_webpage(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         return response.text
#     except requests.RequestException as e:
#         return None

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class BenchmarkScraper:
    def __init__(self):
        # Setup Chrome options for headless browsing
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
    
    def scrape_static_table(self, url, table_selector='table'):
        """
        Scrape data from static HTML tables
        """
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = pd.read_html(response.text)
            
            # Return the first table found or search by selector
            if len(tables) > 0:
                return "".join(tables)
            
            return ""
        except Exception as e:
            return ""

    def scrape_dynamic_table(self, url, table_selector='table', wait_time=10):
        """
        Scrape data from dynamically loaded tables using Selenium
        """
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.get(url)
            
            # Wait for table to load
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
            )
            
            # Get table HTML
            table_html = driver.find_element(By.CSS_SELECTOR, table_selector).get_attribute('outerHTML')
            df = pd.read_html(table_html)[0]
            
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Convert to JSON
            json_data = df.to_json(orient='records')
            
            json_string = json.dumps(json_data)
            
            driver.quit()
            return json_string
        
        except Exception as e:
            print(f"Error scraping dynamic table: {e}")
            if 'driver' in locals():
                driver.quit()
            return None

    def scrape_api_data(self, url):
        """
        Scrape data from API endpoints that return JSON
        """
        try:
            response = requests.get(url)
            data = response.json()
            return pd.json_normalize(data)
        except Exception as e:
            print(f"Error scraping API data: {e}")
            return None

def parse_information_by_gpt(content):
    system_prompt = "You are an extractor who is good at extracting LLM benchmark information from the webpage."

    user_prompt = f"Given the information from the webpage, extract the benchmark information in the format of json list (llm_name, score). You must cover and can only cover all the LLMs shown in the webpage. The llm name should be lowercased and be splitted with '-', such as gpt-4o, claude-3.5-sonnet and llama-3.1-405B. If the LLM is specified by 5-shot or CoT, you need explicitly annotate that like gpt-4o(5-shot), gpt-4o(cot). The webpage information is as below: [{content}]"
 
    model = OpenAI()
    
    response = model.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    # breakpoint()
    response_message = response.choices[0].message.content
    json_data = json.loads(response_message)
    return json_data


def parse_information_by_gemini(content):
    system_prompt = "You are an extractor who is good at extracting LLM benchmark information from the webpage."

    user_prompt = f"""Given the information from the webpage,
extract the benchmark information in the format of json list (llm_name, score). 
You must cover and can only cover all the LLMs shown in the webpage. 
The llm name should be lowercased and be splitted with '-', such as gpt-4o, claude-3.5-sonnet and llama-3.1-405B. 
If the benchmark has multiple scores for each LLM, just use the average score or the overall score. 
All the scores need to be converted into float point values such as 88.9, 99.3, 11.2. The webpage information is as below: [{content}]"""
    # user_prompt = f"Given the information from the webpage, extract the benchmark information in the format of json list (llm_name, score). You must cover and can only cover all the LLMs shown in the webpage. The llm name should be lowercased, such as gpt-4o, claude-3.5-sonnet and llama-3.1-405B. If the LLM is specified by 5-shot or CoT, you need explicitly annotate that like gpt-4o(5-shot), gpt-4o(5-shot,cot). The webpage information is as below: [{content}]"

    import typing_extensions as typing

    class LLM(typing.TypedDict):
        llm_name: str
        score: str
    
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    outputs = model.generate_content(
        system_prompt + user_prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", response_schema=list[LLM]
        )
    )
    result = outputs.candidates[0].content.parts[0].text
    json_data = json.loads(result)
    # breakpoint()
    return json_data

def process_urls(final_results, benchmarks):
    # final_results = {}
    benchmark_scraper = BenchmarkScraper()
    for benchmark in tqdm(benchmarks):
        # static_content = benchmark_scraper.scrape_static_table(benchmark["url"])
        dynamic_content = benchmark_scraper.scrape_dynamic_table(benchmark["url"])
        
        breakpoint()
        # if content:
        #     parsed_content = parse_information_by_gemini(content)
        #     final_results[benchmark["name"]] = parsed_content
        if dynamic_content:
            parsed_content = parse_information_by_gemini(dynamic_content)
            final_results[benchmark["name"]] = parsed_content
        
        # breakpoint()
    return final_results

final_results = process_urls(final_results, benchmarks)

with open("llm_benchmarks.json", "w") as f:
    json.dump(final_results, f, indent=2)

