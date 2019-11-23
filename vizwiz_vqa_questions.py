import json

"""
Vizwiz format to VQA 2.0 format convertor - question annotations only:
This script converts the VizWiz Questions into VQA 2.0 Question annotation format.
The question ID starts from 581922 which is one above 581921 which is the last VQA dataset's question
Output will be stored to a file named: "VizWiz_to_VQA_Questions.json"
Sample i/p format: {"answerable": 1, "image": "VizWiz_train_000000000000.jpg", "question": "What's the name of this product?", "answer_type": "other", "answers": [{"answer_confidence": "yes", "answer": "basil leaves"}, {"answer_confidence": "yes", "answer": "basil leaves"}, {"answer_confidence": "yes", "answer": "basil"}, {"answer_confidence": "yes", "answer": "basil"}, {"answer_confidence": "yes", "answer": "basil leaves"}, {"answer_confidence": "yes", "answer": "basil leaves"}, {"answer_confidence": "yes", "answer": "basil leaves"}, {"answer_confidence": "yes", "answer": "basil leaves"}, {"answer_confidence": "yes", "answer": "basil leaves"}, {"answer_confidence": "yes", "answer": "basil"}]}
Sample o/p format: {"image_id": 458752, "question": "What is this photo taken looking through?", "question_id": 458752000}
"""

#Opening the VizWiz questions file
with open('train.json') as json_file:
    data = json.load(json_file)

#This dictionary contains the results from VizWiz annotations to the VQA 2.0 format
converted_dict={}
converted_dict["questions"]=[]

#The image_id of the first image
base_number=581922

#This dictionary is used to keep track of the last number of the question_id for a given image_id
lastQuestionNumber={}

for i in range(0,len(data)):
		image_id=int(data[i]["image"].split('_')[2].split('.')[0])
		datasample={}
		datasample["image_id"]=581922+image_id
		datasample["question"]=data[i]["question"]

		if datasample["image_id"] in lastQuestionNumber:
			question_id=int(lastQuestionNumber[datasample["image_id"]])+1
			question_id=format(question_id, '03')
			lastQuestionNumber[datasample["image_id"]]=question_id

		else:
			question_id="000"
			lastQuestionNumber[datasample["image_id"]]=question_id

		datasample["question_id"]=int(str(datasample["image_id"])+question_id)

		converted_dict["questions"].append(datasample)

with open('VizWiz_to_VQA_Questions.json', 'w') as outfile:
    json.dump(converted_dict, outfile)
