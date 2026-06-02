import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from developer.training_history import show_training_history
from developer.training_history import get_training_dataframe

FILE1_PATH  = "/home/jose/DSR-UI-Dashbords/Wildfire-Prediction/models/mobilenet_v3_small_training_history.csv"
FILE2_PATH = "/home/jose/DSR-UI-Dashbords/Wildfire-Prediction/models/resnet18_training_history.csv"

PATHS = [FILE1_PATH,FILE2_PATH]

MODELS_DIR = "./models/"
TRAINING_CSV_SUFFIX = "training_history.csv"
EVALUATION_PATH = "./models/evaluation_predictions.csv"

def get_training_dataframe(path_str):
    training_df = pd.read_csv(path_str)
    st.dataframe(training_df)
    return training_df["architecture"][0], training_df

def select_model(model_names):
    option = st.selectbox("Select the model",options=model_names)
    st.write(f"You selected {option}")
    return option


def main():
    st.title("ML Dashboard")

    models_storage = {}
    for path_str in PATHS:
        model_name,model_df=get_training_dataframe(path_str)
        models_storage[model_name] = model_df

    model_names = list(models_storage.keys())
    selected_option =  select_model(model_names)

    st.subheader(f" {selected_option}: Loss during training history")
    selected_df = models_storage[selected_option]

    metric_options = list(selected_df.columns)[2:]
    selected = st.multiselect("Pick your favorite metric:", metric_options)
    st.write("You selected:", selected) 

    st.line_chart(selected_df, x="epoch",y=selected)
 #model_names = []
  ##  for path_str in PATHS:
    #    model_name=get_training_dataframe(path_str)
     #   model_names.append(model_name)

    #selected_option = select_model(model_names)
    #st.text(selected_option)


   # st.header("Training History")
   # training_df = get_training_dataframe(MODELS_DIR+TRAINING_CSV_SUFFIX)
    #df_train = show_training_history(MODELS_DIR, TRAINING_CSV_SUFFIX)
    #st.dataframe(df_train)

    #result_df = training_df.loc[training_df['architecture'] == 'mobilenet_v3_small']
    #train_accuracy = result_df['train_accuracy'].to_numpy()
    #val_accuracy = locresult_df['val_accuracy'].to_numpy()
    

# 1. Create a sample DataFrame
#data = {'inputs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
#df = pd.DataFrame(data)

# 2. Extract the column as a vector
#x_vector = df['inputs'].to_numpy()

# 3. Apply your function to the vector
#y_vector = x_vector**2 + 5

# 4. Plot the vectors
#plt.plot(x_vector, y_vector, marker='o')
#plt.xlabel('Dataframe Column Values')
#plt.ylabel('Function Output')
#plt.show()




    #st.header("Evaluation")
    #evaluation_df = pd.read_csv(EVALUATION_PATH)
    #st.dataframe(evaluation_df)

    #st.header("Resnet18 history")
    #Resnet18_FILE_PATH =  "/home/jose/DSR-UI-Dashbords/Wildfire-Prediction/models/resnet18_training_history.csv"
    #Resnet18_df = pd.read_csv(Resnet18_FILE_PATH)
    #st.dataframe(Resnet18_df)


if __name__ == "__main__":
    main()