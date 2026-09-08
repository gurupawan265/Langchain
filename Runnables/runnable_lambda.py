from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)


# -------------------------
# LLM
# -------------------------

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-2b-it",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 100
    }
)

model = ChatHuggingFace(llm=llm)


# -------------------------
# Prompt
# -------------------------

prompt = PromptTemplate(
    template="Write a short joke about {topic}",
    input_variables=["topic"]
)


# -------------------------
# Parser
# -------------------------

parser = StrOutputParser()


# -------------------------
# Runnable Lambda
# -------------------------

def count_words(text):
    return len(text.split())


word_count = RunnableLambda(count_words)


# -------------------------
# Main Chain
# -------------------------

chain = RunnableSequence(
    prompt,
    model,
    parser,
    RunnableParallel(
        joke=RunnablePassthrough(),
        word_count=word_count
    )
)


# -------------------------
# Invoke
# -------------------------

result = chain.invoke({
    "topic": "AI"
})


print("JOKE:")
print(result["joke"])

print("\nWORD COUNT:")
print(result["word_count"])