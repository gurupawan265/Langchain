from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-2b-it",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 30
    }
)

model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

prompt=PromptTemplate(
    template='answer  the following question \n{question} from the following text -\n {text}',
    input_variables=['text','question']
)
url='https://www.amazon.in/gp/aw/d/B0GCD6WLY7/ref=gwm_qc_btf_Gaming1/?ie=UTF8&pd_rd_w=vfdB1&content-id=amzn1.sym.c74ec136-7792-44e0-8b61-676bf108f362&pf_rd_p=c74ec136-7792-44e0-8b61-676bf108f362&pf_rd_r=STS93AF4SE167YTXSMF6&pd_rd_wg=y66VN&pd_rd_r=60d890ad-341f-4f98-9b9b-7a31bce142f6&ref_=pd_hp_d_r_btf_unk'

loader=WebBaseLoader(url)

docs=loader.load()


chain=prompt|model|parser 

print(chain.invoke({'question':'what is the screen size of the product','text':docs[0].page_content}))

