from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

documents=[
    "Delhi is the capital of india",
    "kolkata is the capital of west bengal",
    "Amaravati is the capital of andhra pradesh"
]

result = embedding.embed_documents(documents)

print(str(result))