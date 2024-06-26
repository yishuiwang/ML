import torch
from torch import nn

data = {
	"answers": ["C", "B", "A"],
	"options": [
		["broke out in North cattle farms", "was controlled as soon as possible", "happened in June,2010", "destroyed almost all cattle in cattle farms"],
		["The government controlled only the two farms.", "The government had more animals killed.", "The government tried to cure the sick animals.", "The government hated to kill animals."],
		["Today's News", "History and Culture", "Entertainment", "Science"]
	],
	"questions": ["Last time foot and mouth disease_.", "How did the government deal with the disease?", "In which part of a newspaper can you most probably read the passage?"],
	"article": "Three more cattle farms in Andong,North Gyeongsang Province,were found to have been infected with the deadly foot-and-mouth disease,Nov.2 2010,Thursday.People fear that livestock farms in other parts of the country could be hit by the virus soon.\nOn Monday,the disease was first detected on two pig farms in Andong,about half a year after the last disease broke out in Korea.A cattle farm in the area also fell victim to the animal disease the following day.\nThe Ministry for Food,Agriculture,Forestry and Fisheries made sure that three more cases of foot-and-mouth disease appeared on Thursday and decided to kill all the animals at the farms and others at nearby places to stop the spread of the virus to other regions.Over 800 cows and pigs within a 500 meter range of the infected farms were killed and buried underground.\n\"Three suspected cases were reported Wednesday,near the pig farms where the first outbreak was reported.The laboratory tests today showed that all three cattle farms were infected with the disease,\" a ministry official said.Two newly infected cattle farms were less than 4 kilometers away from the two pig farms,while the third one was only 2.5 kilometers away.\nThe ministry also said another cattle farm in Andong reported suspected cases of foot-and-mouth disease on its livestock Thursday,indicating the disease will likely continue to spread across the city and possibly beyond.\nThe government has culled more than 33,000 animals near the affected farms Monday alone under its disease prevention program.Additionally,all 84 livestock markets across the nation were closed Wednesday for a period to prevent spread of the disease.\nNo suspected cases have been reported outside of Andong,but the government Thursday decided to destroy an additional 22,000 pigs at two farms in Boryeong,South Chungcheong Province.",
	"id": "high75.txt"
}

arrticle = data['article'][0]
questions = data['questions'][0]
options = data['options'][0]
answers = data['answers'][0]

# 将文章、问题、选项、答案转换为张量




