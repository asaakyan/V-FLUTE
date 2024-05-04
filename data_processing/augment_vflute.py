import openai
import re 
from time import sleep
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

openai.api_key = ''

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gen_contra_claim_memecap(row, init_prompt):
  if row['source'] != 'memecap':
     return row['claim']
  prompt = init_prompt
  prompt += f"{row['claim']}\nExplanation: {row['explanation']}\nOpposite claim:"
  response = openai.chat.completions.create(
    model="gpt-4-0125-preview",
    seed=42,
    stop=["\n\n", "7. Claim"],
    # response_format={ "type": "json_object" },
    messages=[
      # {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
      {"role": "user", "content": prompt}
    ]
  )
  # print(response.choices[0].message.content)
  return response.choices[0].message.content.split("\n")[0].strip()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gen_contra_claim_muse(row, init_prompt):
  if row['source'] != 'muse':
     return row['claim']
  prompt = init_prompt
  prompt += f"{row['claim']}\nExplanation: {row['explanation']}\nNon-sarcastic claim:"
  response = openai.chat.completions.create(
    model="gpt-4-0125-preview",
    seed=42,
    stop=["\n\n", "7."],
    # response_format={ "type": "json_object" },
    messages=[
      # {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
      {"role": "user", "content": prompt}
    ]
  )
  # print(response.choices[0].message.content)
  return response.choices[0].message.content.split("\n")[0].strip()

def adjust_expl(x):
    x = x.replace("directly", "")
    x = x.replace(", thus supporting the claim.", ".")
    x = x.replace(", thus the claim is entailed by the image.", ".")
    x = x.replace(", thus, the claim is entailed by the image.", ".")
    x = x.replace(", which aligns with the provided claim.", ".")
    x = x.replace(", which is related to the claim provided.", ".")
    x = x.replace("Therefore, the claim is entailed by the image.", "")
    x = x.replace("Therefore, the claim contradicts the image.", "")
    x = x.replace("Therefore, the claim is contradicted by the image.", "")
    x = x.replace("The irony in the claim is visually represented",  "The irony is visually represented")
    
    x = x.replace("The claim is entailed through", "Notably, the image displays")
    x = x.replace("The claim is entailed by the image through", "Notably, the image displays")
    x = x.replace("The claim is entailed by the image because it", "Importantly, the image")
    x = x.replace("The claim is entailed by the image as it", "Crucially, the image")
    x = x.replace("The claim is entailed by the image, as it", "Crucially, the image")
    x = x.replace("The claim is entailed by the image as the", "Importantly, the")
    x = x.replace("The claim is entailed by the fact that", "Importantly,")

    x = x.replace("The claim is entailed by the image because", "Crucially,")
    x = x.replace("This image represents the claim because", "Importantly,")
    x = x.replace("The claim is entailed because", "Prominently,")
    
    x = x.replace("The juxtaposition of these two panels visually represents the claim.", "")
    x = x.replace("The claim is conveyed through the", "The image shows")
    x = x.replace("The entailed claim is emphasized by the meme's humorous suggestion", 
                 "The meme humorously suggests")

    x = x.replace("supports the claim", "represents the claim")
    x = x.replace("supporting the claim", "representing the claim")
    x = x.replace("entails the claim", "represents the claim")
    x = x.replace("entailing the claim", "representing the claim")

    x = x.replace(", which representing the claim.", "")
    x = x.replace(", thus representing the claim.", "")
    x = x.replace(", representing the claim provided.", "")

    x = x.replace("The claim is entailed by the image,", "A claim could be conveyed through")
    x = x.replace("The claim is entailed by", "A claim could be conveyed through")
    x = x.replace("the claim", "a claim")
    x = x.replace("The claim", "A claim")                  
    x = x.replace("is entailed by", "could be conveyed by")
    x = x.replace("represents a claim by", "conveys a claim by")

    x = x.replace("  ", " ")

    return x.strip()

def adjust_expl_rest(x):
    x = x.replace("directly", "")
    x = x.replace("Therefore, the claim is entailed by the image.", "")
    x = x.replace("Therefore, the claim contradicts the image.", "")
    x = x.replace("Therefore, the claim is contradicted by the image.", "")
    
    # remove sentence starting with "Therefore, the claim that"
    # use regex
    x = re.sub(r"Therefore, the claim that.*?image\.", "", x)
    x = re.sub(r"Therefore, the image entails.*?", "", x)
    x = re.sub(r"Therefore, the image contradicts.*?", "", x)


    x = x.replace("  ", " ")

    return x.strip()