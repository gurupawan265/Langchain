from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-2b-it",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 300
    }
)

prompt1=PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)
model = ChatHuggingFace(llm=llm)
parser=StrOutputParser()
paralled_chain=RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'LinkedIn':RunnableSequence(prompt2,model,parser)
})

print(paralled_chain.invoke({'topic':'AI'}))