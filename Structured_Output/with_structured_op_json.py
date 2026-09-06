from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 200
    }
)

model = ChatHuggingFace(llm=llm)

json_schema = {
    "title": "Review",
    "type": "object",

    "properties": {
        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List all the key themes discussed in the review"
        },

        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },

        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "Overall sentiment of the review"
        },

        "pros": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List all the advantages mentioned in the review"
        },

        "cons": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List all the disadvantages mentioned in the review"
        },

        "name": {
            "type": "string",
            "description": "Name of the reviewer"
        }
    },

    "required": [
        "key_themes",
        "summary",
        "sentiment"
    ]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say,
it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes
everything lightning fast—whether I'm gaming, multitasking, or editing
photos. The 5000mAh battery easily lasts a full day even with heavy use,
and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches,
though I don't use it often. What really blew me away is the 200MP camera.
The night mode is stunning, capturing crisp, vibrant images even in low
light. Zooming up to 100x actually works well for distant objects, but
anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed
use. Samsung's One UI still comes with bloatware. The $1,300 price tag
is also a hard pill to swallow.

Pros:
Insanely powerful processor
Stunning 200MP camera
Long battery life with fast charging
S-Pen support is unique and useful

Review by Nitish Singh
""")

print(result)