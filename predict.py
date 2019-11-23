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
	parser.add_argument('--model_path', type=str, default = 'Data/train2014/Tri Training 1/Models/model11.ckpt',
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
	parser.add_argument('--data_dir', type=str, default='Data/train2014/Tri Training 1/',
                       help='Data directory')
	parser.add_argument('--question', type=str, default='What is this product?',
                       help='Question')
	
	

	args = parser.parse_args()
	vizwiz_file_path = 'Data/Images'
	vocab_data = data_loader.get_question_answer_vocab(data_dir=args.data_dir)
	qvocab = vocab_data['question_vocab']
	#print(qvocab)
	#print(0/0)
	q_map = { vocab_data['question_vocab'][qw] : qw for qw in vocab_data['question_vocab']}
	vizwiz_questions_path = 'VizWiz_to_VQA_Questions.json'
	with open(vizwiz_questions_path, 'r') as input_file:
		vizwiz_questions = json.loads(input_file.read())
	question_vocab = vocab_data['question_vocab']
	word_regex = re.compile(r'\w+')
	question_ids = np.zeros((1, vocab_data['max_question_length']), dtype = 'int32')
	i = 0
	for file in os.listdir(vizwiz_file_path):
		if file.endswith(".jpg"):
			args.image_path = join(vizwiz_file_path,file)
			args.question = vizwiz_questions['questions'][i]['question']
			i+=1
			print("Image:", args.image_path)
			print("Question:", args.question)

			#fc7_features, image_id_list = data_loader.load_fc7_features(args.data_dir, 'val')
			fc7_features = utils.extract_fc7_features(args.image_path, 'Data/train2014/Tri Training 1/vgg16-20160129.tfmodel')

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

			question_words = re.findall(word_regex, args.question)
			base = vocab_data['max_question_length'] - len(question_words)
			for i in range(0, len(question_words)):
				if question_words[i] in question_vocab:
					question_ids[0][base + i] = question_vocab[ question_words[i] ]
				else:
					question_ids[0][base + i] = question_vocab['UNK']

			ans_map = { vocab_data['answer_vocab'][ans] : ans for ans in vocab_data['answer_vocab']}
			model = vis_lstm_model.Vis_lstm_model(model_options)
			input_tensors, t_prediction, t_ans_probab = model.build_generator()
			sess = tf.InteractiveSession()
			saver = tf.train.Saver()
			saver.restore(sess, args.model_path)

			print(question_ids.shape)
			print(fc7_features.shape)
			print(0/0)
			pred, answer_probab = sess.run([t_prediction, t_ans_probab], feed_dict={
			input_tensors['fc7']:fc7_features,
			input_tensors['sentence']:question_ids,
        		})


			print("Ans:", ans_map[pred[0]])
			answer_probab_tuples = [(-answer_probab[0][idx], idx) for idx in range(len(answer_probab[0]))]
			answer_probab_tuples.sort()
			print("Top Answers")
			for i in range(5):
                                #print(ans_map[answer_probab_tuples[i]])
                                print(ans_map[answer_probab_tuples[i][1] ])

if __name__ == '__main__':
	main()
