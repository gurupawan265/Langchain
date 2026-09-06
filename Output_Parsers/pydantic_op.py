from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Create Hugging Face LLM
# --------------------------------------------------

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-2b-it",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 100
    }
)

model = ChatHuggingFace(llm=llm)


# --------------------------------------------------
# 2. Define Pydantic schema
# --------------------------------------------------

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    city: str = Field(description="City of the person")


# --------------------------------------------------
# 3. Create Pydantic Output Parser
# --------------------------------------------------

parser = PydanticOutputParser(
    pydantic_object=Person
)


# --------------------------------------------------
# 4. Create Prompt
# --------------------------------------------------

template = PromptTemplate(
    template="""
You are a data extraction assistant.

Create information about a fictional person.

Return ONLY a JSON object.
Do NOT return the schema.
Do NOT return explanations.
Do NOT use markdown.

The JSON must have exactly these fields:
"name" - string
"age" - integer
"city" - string

Example:
{{"name": "Rahul", "age": 21, "city": "Delhi"}}

Person name: {name}

{format_instructions}
""",
    input_variables=["name"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# ==================================================
# OLD / WITHOUT CHAIN
# ==================================================

# prompt = template.invoke({
#     "name": "Rahul"
# })
#
# result = model.invoke(prompt)
#
# print("RAW OUTPUT:")
# print(result.content)
#
# final_result = parser.parse(result.content)
#
# print(final_result)
# print(final_result.name)
# print(final_result.age)
# print(final_result.city)


# ==================================================
# NEW / USING CHAIN
# ==================================================

chain = template | model | parser

final_result = chain.invoke({
    "name": "Rahul"
})

print(final_result)

print("Name:", final_result.name)
print("Age:", final_result.age)
print("City:", final_result.city)