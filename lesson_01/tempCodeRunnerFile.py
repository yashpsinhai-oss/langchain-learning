#instantiating and configuring model using ChatGroq class that contains the langchain integration
#not yet called LLM
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv
#we have created an instance of model
chat_model_abstraction=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)
#now we need to use this instance using invoke(input) for sending an input and receiving the response from langchain
#.invoke() lanchains standardized interfaces
response=chat_model_abstraction.invoke("Explain JWT authentication in 3 sentences.")
#we didnt just store the plain text answer but the return value of chat model invocation , langchain chat model return AImessage object 
#which contains the meesage content along with metadata such as response detail and usage information 
print(response)

