import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq

# chat model abstraction using ChatGroq class from the langchain-groq integration
chat_model_abstraction=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)
#system and human starts in lowercase , required by the message-template API
# using chatprompttemplate here , its method from_messages , dont use message class here as we want to deliberately learning template abstraction
prompt_template=ChatPromptTemplate.from_messages(
    [("system","You are a helpful Python tutor."),
    #  when we want to provide our conversation history dynamically to the ChatPromptTemplate, we use MessagePlaceholders 
    # to insert our list of messages(list of langchain messages object) into template along with the latest user question , 
    # both are supplied at run time using invoke on the prompt template and providing
    # our previous conversation history as list of langchain message objects
     MessagesPlaceholder("chat_history"),
     ("human","{question}"),
    ]
)
# on this prompt template we can call invoke method directly , as ChatPromptTemplate also has an invoke method
# which returns ChatPromptValue object, which contains the formatted chat prompt/reusable structure/structured prompt with placeholder, value assigned
chat_prompt_value=prompt_template.invoke({
    # this invoke is called on the prompt_template and not on the chat_model_abstraction
    "chat_history":[("human","My name is Alex."),("ai","Nice to meet you, Alex!")],
    "question":"What is my name?"
}
)

print(chat_prompt_value.to_messages())
# you can inspect its messages using .to_messages() and .to_string() which provide the prompt representation in string format
# output -> [SystemMessage(content='You are a python tutor', additional_kwargs={}, response_metadata={}), HumanMessage(content='Explain dictionary in simple terms', additional_kwargs={}, response_metadata={})]


# Supply the value for the placeholder using the prompt template created above
# formatted_messages=prompt_template.format_messages(topic="dictionary")
# returns a list of messages
# can use this template again and again no need to create new everytime

# Send the formatted messages to the model using the invoke standardized interfaces
# returns AIMessage object
# response=chat_model_abstraction.invoke(prompt_value)

# print(response.content)