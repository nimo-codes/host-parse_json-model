# Import the required libraries 
## flask for the server hosting and request for building api
### jsonify to convert the data to json format
#### pickle to load the ml model into the backend and numpy to use arrays for the inputs of the model
from flask import Flask, jsonify, request ,render_template
import pickle, numpy as np




# creating a flask app and defining a template folder named as template
app = Flask(__name__,template_folder='template')


# defining a function which will take data from the form as input 
def load_model_give_res(data):

    filename = "/Users/jarvis/pymycod/tensorflow_AI/trained_models/lr_model_wine_quality.sav"
    #loading the already saved model
    loaded_model = pickle.load(open(filename, 'rb'))
    ## using values from the form.html as json and asssinging to the different variables
    u1 = float(data['AL'])
    u2 = float(data['VA'])
    u3 = float(data['CA'])
    u4 = float(data['PH'])
    u5 = float(data['SH'])
    u6 = float(data['AL'])
    # using numpy to make a array of this list 
    l1 = np.array([u1,u2,u3,u4,u5,u6])
    # sending  the array as a 1d list to predict the output
    result = loaded_model.predict([l1])
    return result




# creating a app route as home. "/" this by default is a get request so it will render the template from the form.html
@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template("form.html")

# making a new app route called /pred to be redirected when submit button on form.html is clicked
@app.route('/pred', methods = ['GET', 'POST'])
def pred():
     # using post method
     if request.method == "POST": 
        # retrieving data from the form as data in json form
        data = request.form
        # calling the function as recieving wine type as 0 or 1
        name = load_model_give_res(data)
        name = name[0]
        if name == 0:
            return jsonify({"QUALITY":"not good quality wine"})
        else:
            return jsonify({"QUALITY":"greattt quality wine"})

      


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80,debug=True)