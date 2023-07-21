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
    Feature=np.array(book.iloc[:,1])
    Target=np.array(book.iloc[:,2])

    # 生成结构化数据,统一小数点后位数
    dataset = []
    for i in range(len(Feature)):
        diction = {}
        diction['name'] = 'sample_{}'.format(i)
        diction['x'] = round(Feature[i], 4)
        diction['y'] = round(Target[i], 2)
        dataset.append(diction)

    return dataset

# Openai_key 导入
OPENAI_KEY='sk-h...'
os.environ['OPENAI_API_KEY']=OPENAI_KEY

# 读取单特征数据
train=read_data("E:\Research_work\Water_quality\data_set\Paper\Train\js\single\MSC.csv")
test=read_data("E:\Research_work\Water_quality\data_set\Paper\Test\js\single\MSC.csv")

# 撰写Prompt
prompt_template_read_data="""Read the following structured data:{data}."""
# Remember there is a linear functional relationship between 'x' and 'y',it is very important."""

predict_data=[]
prompt_template_predict_data="""You must based on this data's relationship between 'x' and 'y', 
to predict the 'y' in some new samples which you only know their 'x'.
The new sample's x={},each item in the list represent one new sample's 'x'.
Calculate the y value corresponding to each x and write all this y values as a list. The maximum y value not exceed 160.""".format([float(i['x']) for i in test]) #准备测试集

prompt_template=prompt_template_read_data+prompt_template_predict_data

# 构建LLM并预测新样本
llm=OpenAI(temperature=0,max_tokens=512)

Last_prompt=PromptTemplate(template=prompt_template,input_variables=['data'])
read_data_chain=LLMChain(prompt=Last_prompt,llm=llm)
doc=read_data_chain.run(train)
print(doc) #输出测试集的预测结果
