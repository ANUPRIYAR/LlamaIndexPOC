import pandas as pd
from pandasai import SmartDataframe
import config
import openai
from langchain.chat_models import AzureChatOpenAI
from llama_index.query_engine.pandas_query_engine import PandasQueryEngine


# Sample DataFrame
df = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
    "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
})
# Instantiate a LLM
from pandasai.llm import OpenAI

llm = AzureChatOpenAI(azure_endpoint=config.OPENAI_API_BASE,
            openai_api_version=config.OPENAI_API_VERSION,
            deployment_name=config.AZURE_OPEN_AI_DEPLOYMENT_NAME_GPT4,
            openai_api_key=config.OPENAI_API_KEY,
            openai_api_type=config.OPENAI_API_TYPE
                      )
df = SmartDataframe(df, config={"llm": llm})
response = df.chat('Which are the 5 happiest countries?')
print(response)
# print(dir(response))