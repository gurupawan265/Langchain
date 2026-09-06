from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 300
    }
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="""Generate a detailed report on {topic}.
Do not include system messages, user messages, or chat-template tokens.""",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="""Generate a 5-point summary from the following text.

TEXT:
{text}

Return only the 5 summary points.
Do not include system messages, user messages, or chat-template tokens.""",
    input_variables=["text"]
)
parser=StrOutputParser()

chain=prompt1 | model | parser| prompt2 | model |parser

result=chain.invoke({'topic':'Unemployment in India'})

print(result)

chain.get_graph().print_ascii()