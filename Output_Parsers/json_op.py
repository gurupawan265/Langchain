from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-2b-it",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 80
    }
)

model = ChatHuggingFace(llm=llm)
parser=JsonOutputParser()
template=PromptTemplate(
    template="Give me the name,age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt=template.format()
# #print(prompt)  Return a JSON object.

# result=model.invoke(prompt)
# final_result=parser.parse(result.content)
chain=template| model | parser

result=chain.invoke({})
print(result)