import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# chat model abstraction using ChatGroq class from the langchain-groq integration
chat_model_abstraction=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)
#system and human starts in lowercase , required by the message-template API
# using chatprompttemplate here , its method from_messages , dont use message class here as we want to deliberately learning template abstraction
prompt_template=ChatPromptTemplate.from_messages(
    [("system","You are a python tutor"),
     ("human","Explain {topic} in simple terms")
     ]
)

# Supply the value for the placeholder using the prompt template created above
formatted_messages=prompt_template.format_messages(topic="dictionary")

# Send the formatted messages to the model using the invoke standardized interfaces
response=chat_model_abstraction.invoke(formatted_messages)

print(response.content)