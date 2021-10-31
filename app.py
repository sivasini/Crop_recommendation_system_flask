import joblib
from flask import Flask,request,render_template
import numpy as np
import io
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


app = Flask(__name__)

filename = 'finalized_model.sav'
model = joblib.load(filename)
@app.route('/')
def home():
    return render_template('login.html')

database = {'sivasini':'123'}

@app.route('/form_login',methods=['POST','GET'])
def login():
    name1 = request.form['username']
    pwd = request.form['password']
    if name1 not in database:
        return render_template('login.html',info='invalid username')
    else:
        if database[name1] != pwd:
            return render_template('login.html',info='invalid password')
        else:
            return render_template('index.html',name=name1)

@app.route('/predict',methods=['POST',"GET"])
def predict():
    int_features=[int(float(x)) for x in request.form.values()]
 
    final = [np.array(list(int_features))]
    print(final)
    prediction = model.predict(final)
    n = request.form["N"]
    p = request.form["P"]
    df1 = pd.read_csv('Crop_recommendation.csv')
    fig = plt.figure(figsize=(15, 10))
    sns.set_style("white")
    values=[[n,p]]
    data=pd.DataFrame(values,columns=["n","p"])
    dataset = df1.loc[df1.label == prediction[0]]
    '''
    plt.scatter(x=df1.N, y=df1.P, s=50, c='darkkhaki', label='Others')
    plt.scatter(x=data.n, y=data.p, s=50, c='blue', label='Given')
    plt.scatter(x=dataset.N, y=dataset.P, s=50, c='tomato', label='Rice qualities')
    '''
    ax = data.plot(kind='scatter', x='n', y='p', color='blue', label='Given')
    df1.plot(kind='scatter', x='N', y='P',color = 'darkkhaki', label='Others',ax=ax)
    dataset.plot(kind='scatter', x='N', y='P',color='tomato', label='Rice qualities', ax=ax)

    plt.title('Nitrogen and potassium differentiated by humidity', fontweight='bold', fontsize=12)
    plt.xlabel("Nitrogen", fontweight='bold', fontsize=12)
    plt.ylabel("Potassium", fontweight='bold', fontsize=12)
    plt.legend(title='Notes', bbox_to_anchor=(1, 1), fontsize=12, title_fontsize=12)
    plt.savefig('./static/images/new_plot.png')
    #return render_template('untitled1.html', name='new_plot', url='/static/images/new_plot.png')

    return render_template('index.html',pred='The recommended crop is {}'.format(prediction[0].title()),
                           url="./static/images/new_plot.png")


if __name__ ==  '__main__':
    app.run(debug=True)
