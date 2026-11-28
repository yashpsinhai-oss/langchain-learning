from dotenv import load_dotenv
import os

loaded = load_dotenv()
print("Dotenv loaded:", loaded)
print("Key exists:", bool(os.getenv("GROQ_API_KEY")))
print("Working directory:", os.getcwd())

from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate #is the main abstraction here
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
example_prompt=PromptTemplate(#formats each example
    input_variables=["review","sentiment"], #placeholder
    template="Review: {review}\nSentiment: {sentiment}" 
)

# we'll use the fewshotprompttemplate to combine our example list, example prompt template , a prefix explaining the task, a suffix defining where the new input goes
# similar to PromptTemplate input_variables and template we use the parameters for prefix,suffix,exmaples,example_prompt
fewshotprompt_template=FewShotPromptTemplate(
    # Takes the three dictionaries from examples.
    examples=examples,

    # Combines the formatted examples with prefix.
    prefix="Classify the sentiment of the following reviews.",

    # Formats each dictionary using example_prompt.
    example_prompt=example_prompt,

    input_variables=["input"],

    # Adds the suffix, replacing {input} with your new review.
    suffix="""
    Now classify this review:

    Review: {input}
    Sentiment:
    """
)

# now to replace our dynamic placeholder with our new review we will use .invoke on the fewshotprompttemplate
# Returns the final formatted prompt.
print(fewshotprompt_template.invoke({
    "input":"The food was fantastic!"
}))

# till now no LLM or model is connected , once our fewshotprompttemplate output is verified we will connecte the model usng our existing langchain's chat model abstraction
chat_model_abstraction=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# creating chain by connecting multiple langchain components in one pipeline execution using pipe (|) operator which is part of LCEL
# remember this flow prompt template->chat model->output parser , pipe and chain connects this component
chain=fewshotprompt_template | chat_model_abstraction
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