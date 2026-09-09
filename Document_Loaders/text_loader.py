from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-2b-it",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 100
    }
)

model = ChatHuggingFace(llm=llm)
prompt=PromptTemplate(
    template='Write a summary for the following poem -\n {poem}',
    input_variables=['poem']
)

parser=StrOutputParser()
loader=TextLoader('C:\\Users\\LENOVO\\OneDrive\\Desktop\\Langchain\\Document_Loaders\\cricket.txt',encoding='utf-8')

docs=loader.load()

print(type(docs))
print(docs[0].page_content)
print(docs[0].metadata)

chain=prompt|model|parser
print(chain.invoke({'poem':docs[0].page_content}))