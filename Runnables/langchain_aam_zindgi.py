import random


#dummy LLM class->first component
class NakliLLM:

    def __init__(self):#constructor
        print('LLM created')\

    def predict(self,prompt):#second method

        response_list=[
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]    

        return {'response':random.choice(response_list)}

class NakliPromptTemplate:

    def __init__(self,template,input_variables):
        self.template=template
        self.input_variables=input_variables

    def format(self,input_dict):
        return self.template.format(**input_dict)    

class NakliLLMchain:

    def __init__(self,llm,prompt):
        self.llm=llm
        self.prompt=prompt

    def run(self,input_dict):

        final_prompt=self.prompt.format((input_dict))   
        result=self.llm.predict(final_prompt)

        return result['response'] 
    
# llm=NakliLLM()
# print(llm.predict('What is the capital of India'))  

template=NakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length','topic']
)

# prompt=template.format({'length':'short','topic':'india'})
# llm=NakliLLM()
# print(llm.predict(prompt))

llm=NakliLLM()
chain=NakliLLMchain(llm,template)
print(chain.run({'length':'short','topic':'india'}))