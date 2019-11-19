import json

def find_same_first_answer(m1_predictions_path, m2_predictions_path, m3_predictions_path):

    predictions_1 = None
    predictions_2 = None
    predictions_3 = None

    with open(m1_predictions_path, 'r') as p_file:
        predictions_1 = json.loads(p_file.read())['predictions']

    with open(m2_predictions_path, 'r') as p_file:
        predictions_2 = json.loads(p_file.read())['predictions']

    with open(m3_predictions_path, 'r') as p_file:
        predictions_3 = json.loads(p_file.read())['predictions']


    if len(predictions_1) == len(predictions_2) == len(predictions_3):
        i = 0
        aggregated_predictions = []
        for i in range(len(predictions_1)):
            prediction_1 = predictions_1[i]
            prediction_2 = predictions_2[i]
            prediction_3 = predictions_3[i]

            if prediction_1['image_id'] == prediction_2['image_id'] == prediction_3['image_id']:
                if prediction_1['question'] == prediction_2['question'] == prediction_3['question']:
                    if prediction_1['predicted_answer'][0] == prediction_2['predicted_answer'][0] == prediction_3['predicted_answer'][0]:
                        aggregated_predictions.append({
                            'image_id': prediction_1['image_id'],
                            'question': prediction_1['question'],
                            'predicted_answer': prediction_1['predicted_answer'][0],
                            'all_answers': prediction_1['predicted_answer'] + prediction_2['predicted_answer'] + prediction_3['predicted_answer'],
                            'all_confidence': prediction_1['confidence'] + prediction_2['confidence'] + prediction_3['confidence']})

                else:
                    raise Exception('question mismatch, M1: {0}, M2: {1}, M3: {2}'. format(
                        prediction_1['question'], prediction_2['question'], prediction_3['question']))

            else:
                raise Exception('image_id mismatch, M1: {0}, M2: {1}, M3: {2}'. format(
                    prediction_1['image_id'], prediction_2['image_id'], prediction_3['image_id'])) 


        return {'model': 'model_agg', 'predictions': aggregated_predictions}         
    
    else:
        raise Exception('predictions have different lengths, M1: {0}, M2: {1}, M3: {2}'. format(
            len(predictions_1), len(predictions_2), len(predictions_3))