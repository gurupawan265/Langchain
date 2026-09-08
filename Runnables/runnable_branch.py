from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnableLambda

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-2b-it",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 150
    }
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Classify email
classifier_prompt = PromptTemplate(
    template="""
Classify this email as complaint, refund, or general.
Return only the category.

Email: {email}
""",
    input_variables=["email"]
)

classifier = classifier_prompt | model | parser


def clean_category(text):
    text = text.lower()
    if "complaint" in text:
        return "complaint"
    if "refund" in text:
        return "refund"
    return "general"


classifier = classifier | RunnableLambda(clean_category)


# Handling prompts
complaint_prompt = PromptTemplate(
    template="How should we handle this complaint?\n{email}",
    input_variables=["email"]
)

refund_prompt = PromptTemplate(
    template="How should we handle this refund request?\n{email}",
    input_variables=["email"]
)

general_prompt = PromptTemplate(
    template="How should we handle this general query?\n{email}",
    input_variables=["email"]
)


complaint_chain = complaint_prompt | model | parser
refund_chain = refund_prompt | model | parser
general_chain = general_prompt | model | parser


# Branch
branch = RunnableBranch(
    (lambda x: x["category"] == "complaint", complaint_chain),
    (lambda x: x["category"] == "refund", refund_chain),
    general_chain
)


# Complete chain
chain = (
    RunnableParallel(
        email=RunnableLambda(lambda x: x["email"]),
        category=classifier
    )
    | branch
)


print(
    chain.invoke({
        "email": "I purchased the product but it stopped working after two days."
    })
)