from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3.5")

result = model.invoke("What is the capital city of ap")

print(result.content)