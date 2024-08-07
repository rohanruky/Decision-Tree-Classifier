#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gradio as gr
from sklearn import tree

# Training data
features = [[1000, 0, 3], [5000, 0, 1], [20000, 1, 7], [10000, 1, 9], [350, 500, 5]]
labels = ['Cybercrime', 'Cybercrime', 'Through Cybercrime', 'Through Cybercrime', 'Cybercrime']

# Train the model
myClassifier = tree.DecisionTreeClassifier()
myModel = myClassifier.fit(features, labels)

# Prediction function
def predict_cybercrime(f1, f2, f3):
    try:
        f1 = float(f1)
        f2 = float(f2)
        f3 = float(f3)
        prediction = myModel.predict([[f1, f2, f3]])
        return prediction[0]
    except ValueError:
        return "Please enter valid numbers."

# Create Gradio interface
inputs = [
    gr.Number(label="Enter the Money lost"),
    gr.Number(label="Enter if the human was affected as 1"),
    gr.Number(label="Enter the techniques used from 1 to 10")
]

output = gr.Textbox(label="Prediction")

interface = gr.Interface(fn=predict_cybercrime, inputs=inputs, outputs=output, title="Cybercrime Prediction")

interface.launch()


# In[ ]:




