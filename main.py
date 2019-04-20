import json
from .src import emssg, PrepClassifier
from .Preprocessing import preprocessData

if __name__ == '__main__':
    with open('config.json') as json_data_file:
        config = json.load(json_data_file)

    preprocessData.preprocess_europarl(config)
    #emssg.execute_emssg_or_mssg(config)