#instantiating and configuring model using ChatGroq class that contains the langchain integration
#not yet called LLM
from dotenv import load_dotenv
import os

loaded = load_dotenv()
print("Dotenv loaded:", loaded)
print("Key exists:", bool(os.getenv("GROQ_API_KEY")))
print("Working directory:", os.getcwd())
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
#we have created an instance of model
chat_model_abstraction=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)
#now we need to use this instance using invoke(input) for sending an input and receiving the response from langchain
#.invoke() lanchains standardized interfaces
#instead of plain string for the role and message we use the langchain's message classes which provide this roles and messages in structured format
#list is passed to invoke , which the model receives as conversation like structure
#separate message roles allow us to provide instruction to model seperately from the user message
# Change the system message to:
# You are a strict Python tutor. Explain concepts simply and use Python examples.
# You are a pirate. Explain concepts using pirate language.
# human message
# what is jwt authentication
# What is a Python dictionary?
#chat model invocation is stateless from the prespective of chat application, unless you send the earlier conversation again to model it cannot access it
#that differs a chat model and and chat application that maintains conversation history
messages=[SystemMessage(content="You are a pirate. Explain concepts using pirate language.")]
while True:
    user_input=input("HumanMessage : ")

    if user_input.lower() in ['exit','quit']:
        break
    human_message=HumanMessage(content=str(user_input))
    messages.append(human_message)
    response=chat_model_abstraction.invoke(messages)
    messages.append(response)
    print(response.content)
#we didnt just store the plain text answer but the return value of chat model invocation , langchain chat model return AImessage object 
#which contains the meesage content along with metadata such as response detail and usage information 
# print(response)

