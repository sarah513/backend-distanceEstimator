import os
import argparse
from token import ASYNC
import pandas as pd
import time

from playsound import playsound
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

argparser = argparse.ArgumentParser(description='Get predictions of test set')
argparser.add_argument('-m', '--model', help='model json file path')
argparser.add_argument('-w', '--weights', help='model weights file path')
argparser.add_argument('-d', '--data', help='inference data csv file path')
argparser.add_argument('-r', '--results', help='results directory path')

# argparser.add_argument('-i', '--info', help='output of detection')

args = argparser.parse_args()

# parse arguments
#step1
model = args.model
#step2
weights = args.weights

csvfile_path = args.data
# #step3
results_dir = args.results

# step4
# information=args.info

def main():
    # get data
    df_test = pd.read_csv(csvfile_path)
    x_test = df_test[['scaled_xmin', 'scaled_ymin', 'scaled_xmax', 'scaled_ymax']].values
    print('x_test',x_test)
    # print('info',information)
    # standardized data
    scalar = StandardScaler()
    x_test = scalar.fit_transform(x_test)
    scalar.fit_transform((df_test[['scaled_ymax']].values - df_test[['scaled_ymin']]))

    # load json and create model
    json_file = open(model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(weights)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='mean_squared_error', optimizer='adam')
    distance_pred = loaded_model.predict(x_test)

    # scale up predictions to original values
    distance_pred = scalar.inverse_transform(distance_pred)

    # save predictions
    df_result = df_test
    df_result['distance'] = -100000
    
    for idx, row in df_result.iterrows():
        df_result.at[idx, 'distance'] = distance_pred[idx]
        # current_time = time.time()
        # print(current_time)
        # if(distance_pred[idx]<4):
            # playsound('A3TMECN-beep.mp3')
        print(df_result.at[idx, 'distance'])
    df_result.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)


if __name__ == '__main__':
    main()
