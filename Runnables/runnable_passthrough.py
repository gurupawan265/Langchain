from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnablePassthrough,
    RunnableParallel
)

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-2b-it",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 300
    }
)

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke: {text}",
    input_variables=["text"]
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()


# Step 1: Generate the joke
joke_chain = RunnableSequence(
    prompt1,
    model,
    parser
)


# Step 2: Keep the joke AND send it for explanation
chain = joke_chain | RunnableParallel(
    joke=RunnablePassthrough(),
    explanation=RunnableSequence(
        prompt2,
        model,
        parser
    )
)


result = chain.invoke({"topic": "AI"})

print("JOKE:")
print(result["joke"])

print("\nEXPLANATION:")
print(result["explanation"])