# -*- coding: utf-8 -*-
"""Copy of ChatGPT_ABSA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HGq71FJEQot8mAsKwFi95-RnFvBKd0Uo
"""
import openai
import os
import pandas as pd
import time
from tqdm import tqdm

#from dotenv import load_dotenv, find_dotenv
#_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key = "API-KEY" # provide your own api here

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


@retry(wait=wait_random_exponential(min=100, max=1000), stop=stop_after_attempt(6))
def completion_with_backoff(prompt):
    #time.sleep(1)
    return get_completion(prompt, model="gpt-3.5-turbo-16k-0613")

def get_completion(prompt, model="gpt-3.5-turbo-16k-0613"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    #print(response.choices[0].message["content"])
    return response.choices[0].message["content"]

def formatted_to_dataframe(string):
  lines = string.split("\n")
  data = []

  for line in lines:
    #print(line)
    #if len(line) >= 10:
    if len(line) >= 18:
      parts = line.split("/")
      print(len(line))
      print(line)
      aspect = parts[0].strip()
      polarity = parts[1].strip()
      summary = parts[2].strip()
      data.append([aspect, polarity, summary])

  df = pd.DataFrame(data, columns=["Aspect", "Polarity", "Summary"])
  return df

review = """ Had trouble connecting initially. I called customer service.
They rebooted the machine and it worked just fine after that. It is a paid service.
Close to the Papa Murphys Pizza store. Uncovered. Would be accessible at any time day or night.
Well lit parking lot. This is one of the better charging stations.
Seems to be cheaper sometimes n less crowded, maybe? They’re all pretty crowded.
But there’s a Target right there, walking distance, so you can shop n charge if you want.
This is in a residential apt complex with a toddler day care in the middle of it all, just FYI.
There are also other chargers with different plug-ins, I’m not sure what kind, I haven’t checked it out. """
def get_response(review):
  prompt = f"""
  Classify the review and recognize all aspects terms with their corresponding opinion terms and sentiment polarity in [aspect/polarity/summary] format.
  Use the aspects provided: [Accessibility and Accessibility, Amenities and location, Compatibility and Connectivity, Charging speed and efficiency, reliability and maintenance, price and cost, queue and waiting time, customer service, ease of use, payment option, Safety, user interface, mobile app].
  Now I will give you the following examples so that you will be able to better classify the reviews

Review 1: [16 Tesla superchargers and 4 other ev chargers located in the same parking lot as target, ronin, scissors & scotch, yoga six, blue sky, Logan house coffee, crisp & green, mod pizza, I scream gelato, Torchys and chase bank!]
 Amenities and Location/Positive/target, ronin, scissors & scotch, yoga six, blue sky, Logan house coffee, crisp & green, mod pizza, I scream gelato, Torchys and chase bank!]

Review 2:  [16 Tesla superchargers and 4 other ev chargers located in the same parking lot as target, ronin, scissors & scotch, yoga six, blue sky, Logan house coffee, crisp & green, mod pizza, I scream gelato, Torchys and chase bank!]
 Availability and Accessibility/positive/16 Tesla superchargers and 4 other ev chargers

Review 3: [The Tesla Supercharger was fast, convenient, and easy to use! Charging my car was a breeze, and the location was great. Highly recommend for Tesla owners.
 ease of use/positive/convenient, and easy to use

Review 4: ["The Tesla Supercharger was fast, convenient, and easy to use! Charging my car was a breeze, and the location was great. Highly recommend for Tesla owners.]
 charging speed and efficiency/Positive/fast and convenient

Review 5: [This is the worst EA charger. It constantly has one or two chargers (out of 3) not working and there are long lines of cars waiting to charge.]
 reliability and maintenance/Negative/It constantly has one or two chargers (out of 3) not working

Review 6: [This charging station is fairly conveniently located. It's within a parking area for Bank of America, but it's not limited to customer parking. This might be a spot where if you wanted to charge and grab a quick bite nearby, you could, but I generally haven't done that myself. This charging station is generally reliable, although I have had issues a few times where I've had to contact customer support to get going — although to be fair, that's not necessarily an issue isolated to this charging station.]
 customer service/Positive/although I have had issues a few times where I've had to contact customer support to get going

Review 7: [My very first public charging experience - don't think it could have been any better. I hooked up, it identified my car, knew I had a credit through Ford and away it went. Added about 36%, 100 miles, 27kwh in 25 minutes. Will definitely be back.]
 Compatibility and Connectivity/Negative/I hooked up, it identified my car

Review 8: [Debit and credit cards accepted]
 payment option/Positive/Debit and credit cards accepted

Review 9: [$0.34 per kwhr as of 9/22/2022, which is one of cheapest rates on the I-95 corridor currently. Only downside is that Cracker Barrel is only food in walking distance, if only want a snack. But plenty empty chargers so should not have idle chargers if want a full meal]
 price and cost/Positive/one of the cheapest

Review 10: [Problem is that there are lots of chargers but only one is the Chademo that I need. The Nissan Leaf also takes this connection so I had to wait for 20 minutes for it to be free. Meanwhile there were 20 or so unused chargers available.  Maybe better to have a more equal number?  Or even 2 it 3 Chademo?]
 queue and waiting time/Negative/I had to wait for 20 minutes for it to be free

Review 11: [Secure with a guard and security gate]
 safety/Positive/Secure with a guard and security gate

Review 12: [eventually I was able to charge with the touchscreen instead of the app]
 user-interface of charger and mobile app/Positive/able to charge with the touchscreen
  Review text: '''{review}'''
  """
  response = get_completion(prompt)
  return response
response = get_response(review)
print(response)

from google.colab import files
#upload_file = files.upload()

data = pd.read_csv('/content/labeled_text.csv') #upload the file Colab

reviews = data['text'].dropna()

from google.colab import drive
drive.mount('/content/drive')

"""# New Section"""

reviews = reviews.reset_index(drop=True)
reviews

#pd.options.display.max_colwidth = 300

# reviews2 = reviews[5:10]
# rewiews3  = '. '.join(reviews2)
# rewiews3 = f"\'''{rewiews3}\'''"

results = pd.DataFrame(columns=['Aspect', 'Polarity', 'Summary'])
i = 0
for item in tqdm(reviews):
  try:

    response = get_response(item)

    if ('No feedback' not in response) and ('no feedback' not in response):
      # res = response.split(',')
      # print(type(response))
      #res = response.strip('][').split('\n')
      # print(res)
      df = formatted_to_dataframe(response)

      df['Review'] = item
      results = pd.concat([results, df], ignore_index=True)
      i = i + 1
      if i%1000 == 0:
        results.to_csv('results_ABSA.csv', index=False)
  except Exception as e:
        # Handle the specific exception that occurred
      print(f"Error occurred: {e}")
results

"""# New Section"""

results.to_csv('/content/pred_results.csv', index=False)

results.to_csv('/content/drive/MyDrive/results_ABSA.csv', index=False)