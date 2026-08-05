import json
file = "/Users/joshturner/Desktop/LTI Project/MathBot/JSON/lesson1.json"
# with open(file, "r") as f:
#     data = json.load(f)

# chunks = []

# #
# # Learning objectives
# #
# for obj in data["learning_objectives"]:
#     chunks.append({
#         "id": f"week{data['week']}_objective_{obj['tag']}",
#         "text": obj["description"],
#         "metadata": {
#             "week": data["week"],
#             "type": "learning_objective",
#             "tag": obj["tag"]
#         }
#     })

# #
# # Discussions
# #
# for i, discussion in enumerate(data["contents"]["other_material"]):
#     chunks.append({
#         "id": f"week{data['week']}_discussion_{i}",
#         "text": discussion["content_plain"],
#         "metadata": {
#             "week": data["week"],
#             "type": discussion["type"]
#         }
#     })

# #
# # Problems + subproblems
# #
# for problem in data["contents"]["problems"]:

#     context = problem["context_plain"]

#     for sub in problem["subproblems"]:

#         question = sub["plain_text"]["question"]
#         answer = sub["plain_text"]["answer"]

#         chunk_text = f"""
# Problem: {problem['name']}

# Context:
# {context}

# Question:
# {question}

# Answer:
# {answer}
# """

#         chunks.append({
#             "id": f"week{data['week']}_{problem['name']}_{sub['part']}",
#             "text": chunk_text,
#             "metadata": {
#                 "week": data["week"],
#                 "type": "subproblem",
#                 "problem_name": problem["name"],
#                 "part": sub["part"],
#                 "learning_tag": problem["learning_tag"],
#                 "keywords": problem["keywords"]
#             }
#         })

# print(f"Created {len(chunks)} chunks")

# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer(
#     "BAAI/bge-small-en-v1.5"
# )

# embedding = model.encode(chunks[0]["text"])

# import chromadb
# from sentence_transformers import SentenceTransformer

# client = chromadb.Client()

# collection = client.create_collection(
#     name="calculus_notes"
# )

# model = SentenceTransformer(
#     "BAAI/bge-small-en-v1.5"
# )

# for chunk in chunks:
#     embedding = model.encode(chunk["text"]).tolist()

#     collection.add(
#         ids=[chunk["id"]],
#         documents=[chunk["text"]],
#         embeddings=[embedding],
#         metadatas=[chunk["metadata"]]
#     )

#     query = "What does the derivative represent physically?"

# query_embedding = model.encode(query).tolist()

# results = collection.query(
#     query_embeddings=[query_embedding],
#     n_results=3
# )

# print(results["documents"])

# import ollama

# response = ollama.chat(
#     model='qwen3:8b',
#     messages=[
#         {
#             'role': 'user',
#             'content': 'What is slope?'
#         }
#     ]
# )

# print(response['message']['content'])
###################################################################################
###################################################################################
# QDRANT SETUP
#Cluster KEY
# eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MTNjMzM4OWUtNzM1Yi00YWE2LWJiZWMtNzFmNjlmZjEzNzFlIn0.AITo_RUlJFpPiqRINyp2d5CPfe8qy3c2ySQVoOMj34k
# Cluster Endpoint
# https://50b3754e-e428-4aef-b937-13293f0fde89.us-east4-0.gcp.cloud.qdrant.io

# QDRANT SDK
import json
file = "/Users/joshturner/Desktop/LTI Project/MathBot/JSON/lesson1.json"

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document

qdrant_client = QdrantClient(
    url="https://50b3754e-e428-4aef-b937-13293f0fde89.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MTNjMzM4OWUtNzM1Yi00YWE2LWJiZWMtNzFmNjlmZjEzNzFlIn0.AITo_RUlJFpPiqRINyp2d5CPfe8qy3c2ySQVoOMj34k",
)

# print(qdrant_client.get_collections())
import json
file = "/Users/joshturner/Desktop/LTI Project/MathBot/JSON/lesson1.json"

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document


from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import json
import uuid

client = QdrantClient(":memory:")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

with open(file, "r") as f:
    data = json.load(f)

points = []

#
# DEFINITIONS
#
for definition in data["contents"]["definitions"]:

    text = f"""
    Term: {definition['term']}

    Definition:
    {definition['definition_plain']}

    Latex:
    {definition['definition_latex']}
    """

    embedding = model.encode(text).tolist()

    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "week": data["week"],
                "type": "definition",
                "term": definition["term"],
                "text": text
            }
        )
    )

#
# MINI LECTURES / THEOREMS
#
for item in data["contents"]["other_material"]:

    text = f"""
    Type: {item['type']}

    Content:
    {item['content_plain']}

    Latex:
    {item['content_latex']}
    """

    embedding = model.encode(text).tolist()

    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "week": data["week"],
                "type": item["type"],
                "text": text
            }
        )
    )

#
# PROBLEMS
#
for problem in data["contents"]["problems"]:

    for sub in problem["subproblems"]:

        text = f"""
        Problem:
        {problem['name']}

        Context:
        {problem['context_plain']}

        Question:
        {sub['plain_text']['question']}

        Answer:
        {sub['plain_text']['answer']}
        """

        embedding = model.encode(text).tolist()

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "week": data["week"],
                    "type": "problem",
                    "problem_name": problem["name"],
                    "part": sub["part"],
                    "keywords": problem["keywords"],
                    "learning_tag": problem["learning_tag"],
                    "text": text
                }
            )
        )

client.upsert(
    collection_name="calculus",
    points=points
)

query = "How do I find absolute extrema on an open interval?"

query_embedding = model.encode(query).tolist()

results = client.search(
    collection_name="calculus",
    query_vector=query_embedding,
    limit=5
)


results[0].payload["text"]