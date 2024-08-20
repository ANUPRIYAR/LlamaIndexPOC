import pandas as pd
from pandasai import SmartDataframe
import config
from langchain.chat_models import AzureChatOpenAI

def read_xlsx_file():
    # file = raw_input()
    file_path = r"C:\Users\a.m.ramachandran\Downloads\LLM-Web-API-Code-Review.xlsx"
    df = pd.read_excel(file_path, sheet_name='LLM-WEB-API')
    # print(df)
    return df

llm = AzureChatOpenAI(azure_endpoint=config.OPENAI_API_BASE,
            openai_api_version=config.OPENAI_API_VERSION,
            deployment_name=config.AZURE_OPEN_AI_DEPLOYMENT_NAME_GPT4,
            openai_api_key=config.OPENAI_API_KEY,
            openai_api_type=config.OPENAI_API_TYPE
                      )

df = read_xlsx_file()
df = SmartDataframe(df, config={"llm": llm})
response = df.chat('Covert the whole data into dict format and then to json')
print(response)





read_xlsx_file()