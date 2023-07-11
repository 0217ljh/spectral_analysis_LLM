import os
import pandas as pd
import numpy as np
import re
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import PromptTemplate,LLMChain,LLMMathChain
from langchain.chat_models import ChatOpenAI

def read_data(Path):
    # 读取csv数据
    book=pd.read_csv(r"{}".format(Path))
    Feature1=np.array(book.iloc[:,1])
    Feature2= np.array(book.iloc[:, 2])
    Target=np.array(book.iloc[:,3])

    # 生成结构化数据,同时统一小数点后位数
    dataset = []
    for i in range(len(Feature1)):
        diction = {}
        diction['name'] = 'sample_{}'.format(i)
        diction['x'] = [round(Feature1[i], 4),round(Feature2[i], 4)]
    #    diction['x2'] = round(Feature2[i], 4)
        diction['y'] = round(Target[i], 2)
        dataset.append(diction)

    return dataset

# Openai_key 导入
OPENAI_KEY='sk-h...'
os.environ['OPENAI_API_KEY']=OPENAI_KEY

# 读取数据
train=read_data(r"E:\Research_work\Water_quality\data_set\Paper\Train\js\double\BC.csv")
test=read_data(r"E:\Research_work\Water_quality\data_set\Paper\Test\js\double\BC.csv")

# 撰写Prompt
prompt_template_read_data="""Read the following structured data,
each cell in structured data consists of 'name', 'x', 'y', and each 'x' corresponds to two values:
{data}
"""

predict_data=[]
prompt_template_predict_data="""You must based on this data's relationship between 'x' and 'y', 
to predict the 'y' in some new samples which you only know their 'x'.
Here is the 'x' corresponding to the new samples :
{}.
Each item in the list represent one new sample.
Calculate the 'y' value corresponding to each new sample and write all this 'y' values as a list. The maximum y value not exceed 160.""".format([i['x'] for i in test])


prompt_template=prompt_template_read_data+prompt_template_predict_data

# 构建LLM并预测新样本
llm=OpenAI(temperature=0,max_tokens=512)
Last_prompt=PromptTemplate(template=prompt_template,input_variables=['data'])
read_data_chain=LLMChain(prompt=Last_prompt,llm=llm)
doc=read_data_chain.run(train)
print(doc)
