# Main.py


from langchain_community.llms import FakeListLLM, HuggingFaceHub
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os

def use_fake_llm():
    """
    Use the FakeListLLM to generate responses
    """
    
    fake_llm = FakeListLLM(responses=["Hello",
                                      "Hi",
                                      "Ciao!",
                                      "Hola!",
                                      "Bonsoir!",
                                      "Good evening!"])

    prompt = "Hello!"
    print(fake_llm.invoke(prompt))
    print(fake_llm.invoke(prompt))
    print(fake_llm.invoke(prompt))
 

def use_openai_api():
    """
    Use the OpenAI API to generate responses
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
    message = HumanMessage(content="Say Hello in seven different languages.")
    response = llm.invoke([message])
    print(response.content)
    

def use_openai_as_pirate(text):
    """
    Use the OpenAI API to translate text
    """
    template = ChatPromptTemplate([
        ("system", "You are a pirate translator. Translate the text to the way a pirate would say."),
        ("user", "Translate this: {text}")
    ])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)
    response = llm.invoke(template.format_messages(text=text))
    print(response.content)
    
    
    
def translate_using_openai_api(text):
    """
    Use the OpenAI API to translate text
    """
    template = ChatPromptTemplate([
        ("system", "You are an English to French translator. Reject any other language."),
        ("user", "Translate this: {text}")
    ])
    llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(template.format_messages(text=text))
    print(response.content)
    
def translate_using_gemini_api(text):
    """
    Use the Gemini API to translate text
    """
    template = ChatPromptTemplate([
        ("system", "You are an English to French translator. Reject any other language."),
        ("user", "Translate this: {text}")
    ])
    llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(template.format_messages(text=text))
    print(response.content)


def use_hugging_face_hub(text):
    """
    Use the HuggingFaceHub to generate responses
    """
    llm = HuggingFaceHub(
        model_kwargs = {"temperature": 0.5, "max_length": 64},
        repo_id = "google/flan-t5-small", 
        huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    completation = llm.invoke(text)
    print(completation)
    

def translate_using_huggingface_api(text):
    """
    Use the HuggingFace API to translate text
    """
  
    llm = HuggingFaceHub(
        repo_id='Helsinki-NLP/opus-mt-en-fr',
        huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_kwargs={}
    )

    # using the model 
    output = llm.invoke('Can you tell me the capital of russia')

    print(output)
    
    
if __name__ == "__main__":
    
    print("# Using FakeListLLM")
    use_fake_llm()
    
    print("\n # Using OpenAI API")
    use_openai_api()
    
    print("\n # Using OpenAI API to translate")
    translate_using_openai_api("Vamos tocar berimbau na roda de capoeira.")
    
    print("\n # Using OpenAI API to translate to pirate")
    use_openai_as_pirate("I want some alcohol drinks.")
    
    print("\n # Using Gemini API to translate")
    translate_using_gemini_api("How are you doing?")

    print("\n # Using HuggingFaceHub")
    use_hugging_face_hub("In which city is the Eiffel Tower located?")
    
    print("\n # Using HuggingFace API to translate")
    translate_using_huggingface_api("How are you doing?")