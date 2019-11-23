import tensorflow as tf
import vis_lstm_model
import data_loader
import argparse
import numpy as np
import json, os
from os.path import isfile, join
import utils
import re

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default = 'Data/train_2014/COCO_train2014_000000581922.jpg',
                       help='Image Path')
    parser.add_argument('--model_path', type=str, default = 'Data/train2014/Tri Training 3/Models/model49.ckpt',
                       help='Model Path')
    parser.add_argument('--num_lstm_layers', type=int, default=2,
                       help='num_lstm_layers')
    parser.add_argument('--fc7_feature_length', type=int, default=4096,
                       help='fc7_feature_length')
    parser.add_argument('--rnn_size', type=int, default=512,
                       help='rnn_size')
    parser.add_argument('--embedding_size', type=int, default=512,
                       help='embedding_size'),
    parser.add_argument('--word_emb_dropout', type=float, default=0.5,
                       help='word_emb_dropout')
    parser.add_argument('--image_dropout', type=float, default=0.5,
                       help='image_dropout')
    parser.add_argument('--data_dir', type=str, default='Data/train2014/Tri Training 3/',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=200,
                       help='Batch Size')
    parser.add_argument('--question', type=str, default='What is this product?',
                       help='Question')

    args = parser.parse_args()
    #vizwiz_file_path = 'Data/Test'
    solution = dict()
    solution["model"] = "model_3"
    solution["predictions"] = []
    vocab_data = data_loader.get_question_answer_vocab(data_dir=args.data_dir)
    qvocab = vocab_data['question_vocab']
    q_map = { vocab_data['question_vocab'][qw] : qw for qw in vocab_data['question_vocab']}

    vizwiz_questions_path = 'VizWiz_to_VQA_Questions.json'
    with open(vizwiz_questions_path, 'r') as input_file:
        vizwiz_questions = json.loads(input_file.read())

    '''with open('questions_temp.txt','w') as temp_file:
        for i in range(20000):
            #print(vizwiz_questions['questions'][i]['question']
            temp_file.write(vizwiz_questions['questions'][i]['question'])
            temp_file.write('\n')'''
            
        
    question_vocab = vocab_data['question_vocab']
    word_regex = re.compile(r'\w+')
    #question_ids = np.zeros((20000, vocab_data['max_question_length']), dtype = 'int32')
    #fc7_features = np.zeros((2, args.fc7_feature_length))

    print("Reading fc7 features")
    fc7_features, image_id_list = data_loader.load_fc7_features('Data/', 'train')
    print("FC7 features", fc7_features.shape)
    print("image_id_list", image_id_list.shape)
    #print(0/0)
    '''i=0
    for file in os.listdir(vizwiz_file_path):
        if file.endswith(".jpg"):
            args.image_path = join(vizwiz_file_path,file)
            #args.question = vizwiz_questions['questions'][i]['question']
            print("Image:", args.image_path)
            print("Question:", args.question)
            fc7_features[i] = utils.extract_fc7_features(args.image_path, 'Data/vgg16-20160129.tfmodel')
            i += 1'''

    model_options = {
            'num_lstm_layers' : args.num_lstm_layers,
            'rnn_size' : args.rnn_size,
            'embedding_size' : args.embedding_size,
            'word_emb_dropout' : args.word_emb_dropout,
            'image_dropout' : args.image_dropout,
            'fc7_feature_length' : args.fc7_feature_length,
            'lstm_steps' : vocab_data['max_question_length'] + 1,
            'q_vocab_size' : len(vocab_data['question_vocab']),
            'ans_vocab_size' : len(vocab_data['answer_vocab'])
    }
    #question_words = re.findall(word_regex, args.question)
    #base = vocab_data['max_question_length'] - len(question_words)
    '''for no_questions in range(question_ids.shape[0]):
        for i in range(0, len(question_words)):
            if question_words[i] in question_vocab:
                question_ids[no_questions][base + i] = question_vocab[ question_words[i] ]
            else:
                question_ids[no_questions][base + i] = question_vocab['UNK']'''

    ans_map = { vocab_data['answer_vocab'][ans] : ans for ans in vocab_data['answer_vocab']}
    model = vis_lstm_model.Vis_lstm_model(model_options)
    input_tensors, t_prediction, t_ans_probab = model.build_generator()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)
    batch_no = 0
    with open('result3.txt','w') as output_file:
        
        while (batch_no*args.batch_size) < 20000:
            question_ids = np.zeros((args.batch_size, vocab_data['max_question_length']), dtype = 'int32')
            #vizwiz_questions['questions'][i]['question']
            for no_questions in range(question_ids.shape[0]):
                question_formatted = vizwiz_questions['questions'][batch_no*args.batch_size + no_questions]['question']
                question_list = question_formatted.split()
                question_list = question_list[0:20]
                question_formatted = ' '.join(question_list) 
                question_words = re.findall(word_regex, question_formatted)
                base = vocab_data['max_question_length'] - len(question_words)
                for i in range(0, len(question_words)):
                    if question_words[i] in question_vocab:
                        question_ids[no_questions][base + i] = question_vocab[ question_words[i] ]
                    else:
                        question_ids[no_questions][base + i] = question_vocab['UNK']

            fc7 = get_batch(batch_no, args.batch_size, 
                            fc7_features)
            pred, ans_prob = sess.run([t_prediction, t_ans_probab], feed_dict={
                input_tensors['fc7']:fc7,
                input_tensors['sentence']:question_ids,
                })
            
            for i in range(len(pred)):
                current_prediction = dict()
                current_prediction["image_id"] = "VizWiz_train_%.12d.jpg"%(batch_no*args.batch_size + i)
                current_prediction["question"] = vizwiz_questions['questions'][batch_no*args.batch_size + i]['question']

                #output_file.write("Ques:" + vizwiz_questions['questions'][batch_no*args.batch_size + i]['question'])
                answer_list = []
                answer_probab_tuples = [(-ans_prob[i][idx], idx) for idx in range(len(ans_prob[0]))]
                answer_probab_tuples.sort()
                for j in range(5):
                    answer_list.append(ans_map[answer_probab_tuples[j][1]])
                #output_file.write("Ans:" + ans_map[pred[i]])
                current_prediction["predicted_answer"] = answer_list
                #output_file.write("Ans:" + str(answer_list))
                #output_file.write('\n')
                solution["predictions"].append(current_prediction)
                #print("Ans:", ans_map[pred[i]])
                #print('\n')
            batch_no += 1
        output_file.write(json.dumps(solution))

        
def get_batch(batch_no, batch_size, fc7_features):
	'''qa = None
	if split == 'train':
		qa = qa_data['training']
	else:
		qa = qa_data['validation']'''

	si = (batch_no * batch_size)%20000
	ei = min(20000, si + batch_size)
	n = ei - si
	#sentence = np.ndarray( (n, qa_data['max_question_length']), dtype = 'int32')
	#answer = np.zeros( (n, len(qa_data['answer_vocab'])))
	#fc7 = np.ndarray( (n,4096) )

	count = 0

	#for i in range(si, ei):
		#sentence[count,:] = qa[i]['question'][:]
		#answer[count, qa[i]['answer']] = 1.0
		#fc7_index = image_id_map[ qa[i]['image_id'] ]
		#fc7[count,:] = fc7_features[fc7_index][:]
		#count += 1
	
	return fc7_features[si:ei]

if __name__ == '__main__':
    main()
