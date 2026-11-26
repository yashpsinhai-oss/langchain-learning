import os
from dotenv import load_dotenv

load_dotenv()

# chatprompttemplate used when we want structured chat prompt with roles -> return ChatPromptValue
# prompt template when we want a plain text prompt using from_template -> return StringPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# chat model abstraction using ChatGroq class from the langchain-groq integration
chat_model_abstraction=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)
#system and human starts in lowercase , required by the message-template API
# using chatprompttemplate here , its method from_messages , dont use message class here as we want to deliberately learning template abstraction
chat_prompt_template=ChatPromptTemplate.from_messages(
    [("system","You are a python tutor"),
     ("human","Explain {topic} in simple terms")
     ]
)

#we use this when we want to use plain text prompt
prompt_template=PromptTemplate.from_template(
    "Explain {topic} in simple terms"
)

# on this prompt template we can call invoke method directly , as ChatPromptTemplate also has an invoke method
# which returns ChatPromptValue object, which contains the formatted chat prompt/reusable structure/structured prompt with placeholder, value assigned
chat_prompt_value=chat_prompt_template.invoke({
    # this invoke is called on the prompt_template and not on the chat_model_abstraction
    "topic":"dictionary"
}
)

# returns String prompt value object 
prompt_value=prompt_template.invoke({
    # this invoke is called on the prompt_template and not on the chat_model_abstraction
    "topic":"dictionary"
}
)

print(prompt_value.to_string())
print("\n")
print(chat_prompt_value.to_string())
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