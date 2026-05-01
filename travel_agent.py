import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

os.environ["OPENAI_API_KEY"] = "your-api-key"


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# storing the messages
chat_memory = [] 
MAX_TURNS_BEFORE_SUMMARY = 5


main_prompt = ChatPromptTemplate.from_template("""
You are a Travel Agent chatbot.

Rules:
- Answer ONLY travel-related queries (flights, hotels, destinations, itineraries, travel tips).
- If the question is NOT travel-related, respond exactly with:
  "I can’t help with it."

Conversation so far:
{history}

User: {input}
Assistant:
""")


summary_prompt = ChatPromptTemplate.from_template("""
Summarize the following conversation briefly, keeping only important travel-related details:

{history}
""")


def summarize_memory():
    global chat_memory

    history_text = "\n".join(chat_memory)

    summary_chain = summary_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)

    summary = summary_chain.invoke({"history": history_text}).content

    chat_memory = [f"Summary: {summary}"]

    print("\n🧠 Memory summarized!\n")



def chat(user_input):
    global chat_memory

    
    chat_memory.append(f"User: {user_input}")

    history_text = "\n".join(chat_memory)

    chain = main_prompt | llm

    print("\nAssistant:", end=" ", flush=True)

   
    response = chain.invoke({
        "history": history_text,
        "input": user_input
    })

    
    chat_memory.append(f"Assistant: {response.content}")

   
    if len(chat_memory) >= MAX_TURNS_BEFORE_SUMMARY * 2:
        summarize_memory()



if __name__ == "__main__":
    print("✈️ Travel Agent Chatbot (type 'exit' to quit)\n")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        chat(user_input)