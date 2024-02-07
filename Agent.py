#!/usr/bin/env python
# coding: utf-8

# # Chatbot
This is an example of an AI-powered chatbot that can help a student choose a master program. 
It is based on the "gpt-3.5-turbo-1106" model from Open AI and it can use a database, in the form of a .txt file.
The database contains information about the student.
It uses 2 tools, one for accesing the database ("retriever_tool") and one for web searching("search_tool").
It also has a chat history, such that the student can have a conversation with the bot.
The chat history resets after closing the program. You can copy paste the chat history and save it for other sessions if you want.

Do not forget to add the Open AI and Serper API Keys !!!
Do not forget to install all the necessary modules, like Langchain. 
# ## Definitions

# In[1]:


import json
import requests

# Change this
#openai.api_key = "YOUR_OPEN_AI_KEY"   # https://platform.openai.com/docs/overview
#serper_api_key = "YOUR_SERPER_KEY"    # https://serper.dev/dashboard

#It is better for security purposes if you set the above keys as  enviroment variables


# ## Define LLM

# In[2]:


from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)


# ## Define the Database (information about the student)

# In[3]:


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader



from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("", glob="Description_student.txt", loader_cls=TextLoader) # the .txt file should contain information about the student

docs = loader.load()

print(docs[0].page_content)


# In[5]:


from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()


# In[6]:


from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)


# In[7]:


from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
#retrieval_chain = create_retrieval_chain(retriever, document_chain)


# # Define th Tools

# ### 1. Retriver (information about the student)

# In[8]:


from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "student_information",
    "Information about the student. Each time you are asked a question you must take into acount the description of the student, so use this tool!",
)


# ### 2. Web searching

# In[9]:


#@tool
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


# In[10]:



from langchain.agents import initialize_agent, Tool
search_tool = Tool(
        name="Search",
        func=search,
        description="You should always use this tool, as your answers must be based on real web searches"
    )


# In[11]:


tools = [retriever_tool, search_tool]


# ## Prompt

# In[12]:


from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
import typing
from langchain_core.messages import AIMessage, HumanMessage, ChatMessage, SystemMessage, FunctionMessage, ToolMessage


temp= """You are an helful agent talking to a student and helping him/her in choosing a master program. 
         Take the following steps into acount: 
          1)You use the description of the student and you search on the internet for appropriate master programs.
          2)Always do a web search for the master programs and write the link where you got the information from.
          3) Do not make stuff up and do not invent links
          4)You ask the student questions such that you can find a program suited for him more easily."""

prompt = ChatPromptTemplate(
    input_variables=[
        'agent_scratchpad',
        'input'
    ],
    input_types={
        'chat_history': typing.List[
            typing.Union[
                AIMessage,
                HumanMessage,
                ChatMessage,
                SystemMessage,
                FunctionMessage,
                ToolMessage
            ]
        ],
        'agent_scratchpad': typing.List[
            typing.Union[
                AIMessage,
                HumanMessage,
                ChatMessage,
                SystemMessage,
                FunctionMessage,
                ToolMessage
            ]
        ]
    },
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template= temp
            )
        ),
        MessagesPlaceholder(
            variable_name='chat_history',
            optional=True
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['input'],
                template='You answer the question: "{input}".'
            )
        ),
        MessagesPlaceholder(
            variable_name='agent_scratchpad'
        )
    ]
)
print(prompt)


# ## Create the agent

# In[13]:


from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

# Get the prompt to use - you can modify this!
#prompt = hub.pull("hwchase17/openai-functions-agent")


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#prompt


# ## Automatic memory

# In[14]:


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

message_history = ChatMessageHistory()

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# ## Testing

# In[15]:


"""
The following parts are comented such that we don't waste tokens every time we run the program.
Be awere that the maximum token size is not very big. If I write "Give me 3 recomandations of masters!" with 3 instead of 2,
the bot cannot handle it.
"""


# In[16]:


"""
agent_with_chat_history.invoke(
    {"input": "Give me 2 recomandations of masters!"},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}},
)
"""


# In[17]:


"""
agent_with_chat_history.invoke(
    {"input": "Elaborate more on the second one!"},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}},
)
"""


# In[18]:


message_history


# In[ ]:




