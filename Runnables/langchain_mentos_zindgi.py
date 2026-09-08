import random
from abc import ABC, abstractmethod


# Base Runnable
class Runnable(ABC):

    @abstractmethod
    def invoke(self, input_data):
        pass


# -------------------------
# Dummy LLM
# -------------------------
class NakliLLM(Runnable):

    def __init__(self):
        print("LLM created")

    def invoke(self, prompt):

        response_list = [
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence"
        ]

        return {
            "response": random.choice(response_list)
        }

    def predict(self, prompt):

        response_list = [
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence"
        ]

        return {
            "response": random.choice(response_list)
        }


# -------------------------
# Dummy Prompt Template
# -------------------------
class NakliPromptTemplate(Runnable):

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_data):
        return self.template.format(**input_data)

    def format(self, input_dict):
        return self.template.format(**input_dict)


# -------------------------
# Dummy String Output Parser
# -------------------------
class NakliStrOutputParser(Runnable):

    def invoke(self, input_data):

        # LLM returns:
        # {"response": "Delhi is the capital of India"}

        return input_data["response"]


# -------------------------
# Runnable Connector
# -------------------------
class RunnableConnector(Runnable):

    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):

        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data


# -------------------------
# Create Components
# -------------------------

template = NakliPromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=["length", "topic"]
)

llm = NakliLLM()

parser = NakliStrOutputParser()


# -------------------------
# Create Chain
# -------------------------

chain = RunnableConnector([
    template,
    llm,
    parser
])


# -------------------------
# Invoke Chain
# -------------------------

result = chain.invoke({
    "length": "long",
    "topic": "India"
})

print(result)