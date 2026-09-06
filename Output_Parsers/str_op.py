from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.1
    }
)

model = ChatHuggingFace(llm=llm)

# 1st prompt → detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# 2nd prompt → summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text.\n{text}",
    input_variables=["text"]
)

# First LLM call
prompt1 = template1.invoke({
    "topic": "black hole"
})

result = model.invoke(prompt1)

print("FIRST RESULT:")
print(result.content)

# Second LLM call
prompt2 = template2.invoke({
    "text": result.content
})

result1 = model.invoke(prompt2)

print("\nSECOND RESULT:")
print(result1.content)