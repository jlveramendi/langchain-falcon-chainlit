from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os

from dotenv import load_dotenv
import chainlit as cl

# Load environment variables from .env file
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

repo_id = "tiiuae/falcon-7b-instruct"




custom_prompt_template = """You are a helpful AI assistant and provide the answer for the question asked politely.

Question: {question}

Answer: Let's think step by step. """




def load_llm():
    llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                        repo_id=repo_id, 
                        model_kwargs={"temperature":0.7, "max_new_tokens":700})

    return llm

def falcon_bot():
    # Instantiate the chain for that user session
    llm = load_llm()
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['question'])
    llm_chain = LLMChain(prompt=prompt, llm=llm)#, verbose=True)

    return llm_chain

"""  KeyError: 'langchain_factory' :  En la API-REFERENCE ya no figura por ello lanza error:  https://docs.chainlit.io/api-reference/on-chat-start

@cl.langchain_factory
def langchain_factory():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain
"""






@cl.on_chat_start
async def start():
    llm_chain = falcon_bot()

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)



@cl.on_message
async def main(message):
    llm_chain = cl.user_session.get("llm_chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer= True, answer_prefix_tokens= ["FINAL","ANSWER"]
    )
    cb.answer_reached = True
    res = await llm_chain.acall(message.content, callbacks=[cb])
    #res = await llm_chain.acall(message, callbacks=[cb])     # for previous langchain versions
    #answer = res["result"]
    #sources = res["source_documents"]

    #if sources:
    #    answer += f"\nSources---:" + str(sources)
    #else:
    #    answer += f"\nNo Sources Found"

    await cl.Message(content=res["text"]).send()
    #await cl.Message(content=answer).send()





"""
@cl.on_message
async def main(message: cl.Message):
#async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # jv:
    #
    #cb = cl.AsyncLangchainCallbackHandler(
    #    stream_final_answer= True, answer_prefix_tokens= ["FINAL","ANSWER"]
    #)
    #cb.answer_reached = True
    

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    #res = await llm_chain.acall(message, callbacks=[cb])
    #answer = res["result"]

    # Do any post processing here

    # Send the response
    await cl.Message(content=res["text"]).send()
    # await cl.Message(content=answer).send()
"""



"""
https://docs.chainlit.io/examples/qa
"""
