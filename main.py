# Importing dependencies
import os
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import warnings

warnings.simplefilter('ignore')
# Loading the apis
load_dotenv()

# Setting up the models
llm1 = ChatGroq(groq_api_key = os.getenv('GROQ_API_KEY'),
                model_name = "llama-3.1-8b-instant",
                temperature = 0.8)

llm2 = ChatGroq(groq_api_key = os.getenv('GROQ_API_KEY'),
                model_name = "llama-3.1-8b-instant",
                temperature = 0.8)

# Setting up the prompts
prompt1 = PromptTemplate(
    template = ("You are a {language} developer and you need to help the user to generate specific codes to solve the problem given by the user.\n"
                "You need to keep in mind the TIME SPACE COMPLEXITY of the code and make it as precise as possible.\n"
                "After you have presented a answer you will be cross questioned by another AI and it will find flaws in your code. The lesser code it finds the better you performed.\n"
                "You need to make sure you solve the found problems by the AI effectively\n"
                "You will be provided with your evaluation in ONLY the format of a dictionary file\n"
                "Example Format : \n"
                "{{'problems' : 'If there are any problems that needs to be fixed provide them in the form of a list.', 'proceed' : 'A boolean signal if you should present your final answer or not.'}} \n"
                "You also need to answer in ONLY the following dictionary format :\n"
                "You have to also make sure that the dictionary you are providing are single string and not multiline strings.\n"
                "{{'problems resolved' : 'The problems you resolved', 'final answer' : 'This would contain the final answer otherwise NULL'}}"
                ),
    input_variables = ['language'] )

prompt2 = PromptTemplate(
    template = ("You are a senior {language} developer and you need to evaluate the code that has been given to you.\n"
                "You need to judge the code on the basis of TIME SPACE COMPLEXITY and its accuracy.\n"
                "You need to give feedback about the problems in the code in a very concise but explanatory way as possible and make sure that any beginner can also understand your feedback and improve.\n"
                "Keep the feedback and checking simple and no need to be rigid about a specific error, keep your major focus on any sort of syntax error.\n"
                "Provide the answer only in the SPECIFIED FORMAT no need to add any unnecessary remark or quotes outside the specified format. No need to tell that you are providing a dictionary file just return your response.\n"
                "You need to answer in a dictionary format.\n"
                "The format of your response should be as the dictionary file given below :\n"
                "Make sure you provide this dictionary as a single line string not multiline.\n"
                '{{"problems" : "This will contain all the issues or errors that you found in the code. No need to give the solutions.", "proceed" : "This will have either True or False , True if no more fix required and False if fixes still required."}}\n'
                "You can give feedback a maximum of 5 times and after that you need to give turn 'proceed' to True. So, keep resolving syntax errors priority."
                "If you are not sure about some topic no need to give any feedback and return the JSON with 'proceed' : True."),
    input_variables = ['language'])

# Setting up the conversation memories
llm1_memory = ConversationSummaryBufferMemory(llm = llm1)   # main chatbot memory.
eval_memory = ConversationSummaryBufferMemory(llm = llm2)   # evaluation model memory.

 # Setting up the conversation chain memory.
user_conversation = ConversationChain(llm = llm1,
                                 verbose = False,
                                 memory = llm1_memory)
eval_conversation = ConversationChain(llm = llm2,
                                      verbose = False,
                                      memory = eval_memory)

def query_answerer(query : str):
    """Function which will take the query from the user and provide the final answer."""
    res = user_conversation.predict(input = query)
    for k in range(5):
        eval_res = eval_conversation.predict(input = res)
        print(eval_res)
        if eval(eval_res)['proceed']:
            res = user_conversation.predict(input=eval_res)
            break
        else:
            res = user_conversation.predict(input=eval_res)
            eval_res = eval_conversation.predict(input = res)
    eval_memory.chat_memory.clear()
    print(res)
    return eval(res)['final answer']





