from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Create the Hugging Face LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 500
    }
)

# Convert it into a LangChain chat model
model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
]

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Add user's message
    chat_history.append(HumanMessage(content=user_input))

    # Send complete conversation
    result = model.invoke(chat_history)

    # Add AI's response
    chat_history.append(AIMessage(content=result.content))

    print("AI:", result.content)

print(chat_history)