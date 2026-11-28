from dotenv import load_dotenv
import os

loaded = load_dotenv()
print("Dotenv loaded:", loaded)
print("Key exists:", bool(os.getenv("GROQ_API_KEY")))
print("Working directory:", os.getcwd())

# fewshotchatmessageprompttemplate uses chatprompttemplate,suitable for conversation/chat models,creates chat messages
# generates few shot examples as chat messages Example: HumanMessage,AIMessage
# notice how we can see the Human and AI instead of plain text conversation now
from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate #is the main abstraction here like fewshotprompttemplate in previous
#  our architecture will be Examples->chatprompttemplate->fewshotchatmessageprompttemplate->chatprompttemplate(outer template)->chatmodel
# importing our messages class
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langchain_groq import ChatGroq
# instead of manually writing all examples in single prompt we maintain and seperate the examples
# and let langchain create the final prompt
# examples(input output pairs example) -> example prompt(defines how an individual example should look) ->few shot prompt template(examples and suffix combined) -> final prompt(combines all our examples into one final prompt)
examples=[
    {
        "review":"The food was terrible.",
        "sentiment":"Negative"
    },
    {
        "review":"I absolutely loved the experience.",
        "sentiment":"Positive"
    },
    {
        "review":"The service was okay.",
        "sentiment":"Neutral"
    }
]
# we're defining the formatting rules, not manually extracting the data.
# so no loop is needed for this , we can do this only using the PromptTemplate class
# LangChain fills these placeholders when you format the template.
# FewShotPromptTemplate will iterate through the examples list and pass each dictionary to example_prompt automatically., to format each example dictionary
example_prompt=ChatPromptTemplate.from_messages([#formats each example
    # You also don't need to explicitly specify input_variables here. LangChain can infer them from the placeholders.
    # input_variables=["review","sentiment"], #placeholder
    # template="Review: {review}\nSentiment: {sentiment}" 
    ("human","{review}"),
    ("ai","{sentiment}")
])

# FewShotChatMessagePromptTemplate has one responsibility: generate the example chat messages.
# It doesn't manage the overall conversation or the latest user question.
fewshotprompt_template=FewShotChatMessagePromptTemplate(
    # Takes the three dictionaries from examples.
    examples=examples,

    # Combines the formatted examples with prefix.
    # prefix="Classify the sentiment of the following reviews.",  not supported here

    # Formats each dictionary using example_prompt.
    example_prompt=example_prompt,

    # not supported
    # input_variables=["input"],

    # Adds the suffix, replacing {input} with your new review.
    # not supported
    # suffix="""
    # Now classify this review:

    # ("human","{input}"),
    # ("ai","")
    # """
)

# Where do prefix and suffix go?
# We introduce another ChatPromptTemplate that wraps the few-shot examples.
# overall structure system message->fewshotchatmessageprompttemplate->latest humam message
# the outer ChatPromptTemplate controls where those examples appear in the complete conversation.
outerchatprompt_template=ChatPromptTemplate.from_messages([
    ("system","You are a sentiment classifier."),
    # ChatPromptTemplate.from_messages() accepts different message representations, including:
    # System, Human and AI message tuples,Individual message templates.Other prompt templates, including FewShotChatMessagePromptTemplate.
    # When LangChain encounters fewshotprompt_template, it expands it into the formatted example messages.
    fewshotprompt_template,
    ("human","{input}")
])

# now to replace our dynamic placeholder with our new review we will use .invoke on the outerpromottemplate
# Returns the final formatted prompt.
print(outerchatprompt_template.invoke({
    "input":"The food was fantastic!"
}))

# till now no LLM or model is connected , once our outerprompttemplate is verified we will connecte the model using our existing langchains chat model abstraction
chat_model_abstraction=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# creating chain by connecting multiple langchain components in one pipeline execution using pipe (|) operator which is part of LCEL
# remember this flow prompt template->chat model->output parser , pipe and chain connects this component
chain=outerchatprompt_template | chat_model_abstraction
# now since we have connected different components using chain but have to perform invoke for both so we can call invoke on the chain itself
# output for 1st operation becomes input for the second operation
response=chain.invoke({
    "input":"The food was fantastic!"
})

# | pipe operator connects component
# .invoke() executes component 
# LCEL langchain expression language
# chain is connected sequence of components
print(response.content)