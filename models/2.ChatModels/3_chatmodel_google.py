import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


result = model.invoke("What is the capital city of Andhra Pradesh?")

# Print the text response
print(result.content)
