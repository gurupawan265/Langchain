from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import re


# ============================================================
# LLM
# ============================================================

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-2b-it",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 20,
        "temperature": 0.1,
        "return_full_text": False
    }
)


model = llm


# ============================================================
# Pydantic Model
# ============================================================

class Feedback(BaseModel):

    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )


# ============================================================
# CLASSIFIER
# ============================================================

prompt1 = PromptTemplate(
    template="""Classify this feedback as positive or negative.

Feedback: {feedback}

Answer with ONLY one word:
positive
or
negative

Answer:""",
    input_variables=["feedback"]
)


classifier_chain = prompt1 | model | StrOutputParser()


# ============================================================
# Convert LLM output into Pydantic object
# ============================================================

def classify_feedback(x):

    result = classifier_chain.invoke({
        "feedback": x["feedback"]
    })

    result = result.lower().strip()

    if "positive" in result:
        sentiment = "positive"

    elif "negative" in result:
        sentiment = "negative"

    else:
        raise ValueError(
            f"Could not determine sentiment from: {result}"
        )

    return {
        "feedback": x["feedback"],
        "sentiment": Feedback(sentiment=sentiment)
    }


# ============================================================
# Positive response
# ============================================================

prompt2 = PromptTemplate(
    template="""Write an appropriate response to this positive feedback:

{feedback}

Return only the response.""",
    input_variables=["feedback"]
)


# ============================================================
# Negative response
# ============================================================

prompt3 = PromptTemplate(
    template="""Write an appropriate response to this negative feedback:

{feedback}

Return only the response.""",
    input_variables=["feedback"]
)


# ============================================================
# Conditional Branch
# ============================================================

branch_chain = RunnableBranch(

    (
        lambda x: x["sentiment"].sentiment == "positive",
        prompt2 | model | StrOutputParser()
    ),

    (
        lambda x: x["sentiment"].sentiment == "negative",
        prompt3 | model | StrOutputParser()
    ),

    RunnableLambda(
        lambda x: "Could not determine sentiment."
    )
)


# ============================================================
# Complete chain
# ============================================================

chain = (
    RunnableLambda(classify_feedback)| branch_chain
)


# ============================================================
# Test
# ============================================================

result = chain.invoke(
    {
        "feedback": "This is terrible phone"
    }
)

print(result)


# ============================================================
# Graph
# ============================================================

chain.get_graph().print_ascii()