from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
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

parser=StrOutputParser()

prompt=PromptTemplate(
    template='answer  the following question \n{question} from the following text -\n {text}',
    input_variables=['text','question']
)


loader=DirectoryLoader(
    path='C:\\Users\\LENOVO\\OneDrive\\Desktop\\Langchain\\Document_Loaders\\books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
    )

docs=loader.load()


chain=prompt|model|parser 

print(chain.invoke({'question':'','text':docs[0].page_content}))


