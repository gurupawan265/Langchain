from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-2b-it",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 80
    }
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(
        name="fact_1",
        description="fact 1 about the topic"
    ),
    ResponseSchema(
        name="fact_2",
        description="fact 2 about the topic"
    ),
    ResponseSchema(
        name="fact_3",
        description="fact 3 about the topic"
    ),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 facts about the {topic}\n{format_instruction}",
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

prompt = template.invoke({
    "topic": "black hole"
})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)