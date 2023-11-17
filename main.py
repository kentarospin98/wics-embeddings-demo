from sentence_transformers import SentenceTransformer
import numpy as np
from sentence_transformers.util import cos_sim
from fastapi import FastAPI
from redis.asyncio import Redis
import json
from dotenv import load_dotenv
import aiohttp
import os
import re

load_dotenv()

sess = aiohttp.ClientSession()
mpnet = SentenceTransformer("all-mpnet-base-v2")
app = FastAPI()
r = Redis()

AI21_API_KEY = os.getenv("AI21_API_KEY")

data = None

prompt = """You are a Bot that answers questions based on the articles provided. You do not answer any questions that are not relevant to the articles, and you provide no information that is not in the articles.
All your answered are based solely on the articles. If the information needed to answer the question is not in the articles, you say you don't know the answer.
You are polite, and well mannered.

The following is an example on how to answer question.

Articles:

Division is the opposite of multiplication. If 3 groups of 4 make 12 in multiplication, 12 divided into 3 equal groups give 4 in each group in division.

The main goal of dividing is to see how many equal groups are formed or how many are in each group when sharing fairly. 

In the above example, to divide 12 donuts into 3 similar groups, you will have to put 4 donuts in each group. So, 12 divided by 3 will give the result 4.

###

Question: What is division
Thoughts: The article directly answers the question, that Division is the opposite of multiplication, and that the goal of division is to see how many equal groups are formed when sharing fairly.
Answer: Division is the process of dividing a whole into parts fairly, and seeing how many equal groups are formed. It is the opposite of multiplication.

###

Question: Who are you?
Thoughts: This question is not relevant to the articles, so I will not answer it.
Answer: I cannot answer any questions that are not in my dataset.

###

Question: Can you give an example for dividing?
Thoughts: The article provided has an example of division, that is 3 groups of 4 make 12, then 12 divided into 3 equal groups gives 4 in each group.
Answer: If you wanted to divide 12 by 3, you can think of it has if you divide 12 into 3 equal groups, you will get 4 in each group. So 12 divided by 3 is 4.

###

Question: Who was frodo baggins?
Thoughts: This question is not relevant to the articles, so I will not answer it.
Answer: I do not know the answer to that question.

============

Now it's your turn to answer question.

Articles:
"""

with open("dataset.json") as f:
    data = json.loads(f.read())

for d, embedding in zip(data, mpnet.encode([d["text"] for d in data])):
    d["embedding"] = embedding


async def ai21_completion(prompt, model="j2-ultra"):
    url = f"https://api.ai21.com/studio/v1/{model}/complete"
    payload = {
        "prompt": prompt,
        "numResults": 1,
        "maxTokens": 80,
        "minTokens": 0,
        "temperature": 0.7,
        "topP": 1,
        "topKReturn": 0,
        "frequencyPenalty": {
            "scale": 0,
            "applyToWhitespaces": True,
            "applyToPunctuations": True,
            "applyToNumbers": True,
            "applyToStopwords": True,
            "applyToEmojis": True,
        },
        "presencePenalty": {
            "scale": 0,
            "applyToWhitespaces": True,
            "applyToPunctuations": True,
            "applyToNumbers": True,
            "applyToStopwords": True,
            "applyToEmojis": True,
        },
        "countPenalty": {
            "scale": 0,
            "applyToWhitespaces": True,
            "applyToPunctuations": True,
            "applyToNumbers": True,
            "applyToStopwords": True,
            "applyToEmojis": True,
        },
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {AI21_API_KEY}",
    }
    async with sess.post(url, json=payload, headers=headers) as res:
        if res.ok:
            return await res.json()
    return None


@app.get("/")
async def read_root():
    return {"Hello": await r.get("test")}


@app.post("/query")
async def query(query: str):

    # Encode the query
    query_emb = mpnet.encode(query)

    # Compare the query embeddings with the dataset embeddings
    sim_scores = [
        x.item() for x in cos_sim(query_emb, [d["embedding"] for d in data])[0]
    ]

    # Fetch the best matched article
    idx = sim_scores.index(max(sim_scores))
    rel_article = data[idx]

    # Insert it into the prompt
    stitch = prompt + rel_article["text"] + "\n###\n\nQuestion:" + query + "\nThoughts:"

    # Generate the answer
    completion = await ai21_completion(stitch)
    if completion is None:
        return {"answer": "Couldn't answer", "why": "AI Failure."}

    text = completion["completions"][0]["data"]["text"]

    # Parse the output
    matches = re.match("([\s\S]*)\nAnswer:([\s\S]*)",  text)
    if matches is None:
        return {"answer": "Couldn't answer", "why": "AI Failure."}

    return {"answer": matches[2], "why": matches[1], "citation": rel_article["link"]}

@app.post("/classify")
async def classify(query: str, labels: list[str]):
    print(query)

    # Encode the query and labels
    query_emb = mpnet.encode(query)
    label_embs = mpnet.encode(labels)
    
    # Compare the query embeddings with the label embeddings
    sim_scores = [
        x.item() for x in cos_sim(query_emb, label_embs)[0]
    ]

    # Return the best match
    idx = sim_scores.index(max(sim_scores))
    return [labels[idx], sim_scores]