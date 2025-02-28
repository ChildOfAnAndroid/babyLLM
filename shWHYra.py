#Charis Cat 2025

#from transformers import AutoModelForCausalLM, AutoTokenizer

#model_name = "meta-llama/Llama-3.1-8B"

# Download model & tokenizer (MPS optimized)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map="mps")


#ASSIGN
from transformers import pipeline 

#transformer = "mistralai/Mistral-7B-Instruct-v0.1"
transformer = "meta-llama/Llama-3.1-8B"

#generator = pipeline("text-generation", model="meta-llama/Llama-3.1-8B", device="mps") #mps for apple m3 chip device="mps"
generator = pipeline("text-generation", model=transformer, device=-1)

#TAKE INPUT
def userInputFUNK():
    userInput = input("You: ")
    return userInput

#GENERATE RESPONSE
def generateResponse(userInput):
    response = generator(userInput,
                         max_new_tokens=200)[0]["generated_text"]
                         #max_length = 50)
    print(f"whyTho: {response}")

#MAIN
def mainChat():
    while True:
        userInput = userInputFUNK()
        if userInput == "exit":
            break
        generateResponse(userInput)

#RUN
mainChat()




