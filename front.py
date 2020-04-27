from flask import Flask, render_template, url_for, redirect, request, flash
import akn_nlp
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/submit', methods = ['POST'])
def submit():
    if request.method == 'POST':
        str = request.form['t1']
        fp = open("testing.csv","a")
        fp.writelines(str)
        fp.write("\n")
        fp.close()       
        return redirect( url_for('index'))


@app.route('/login')
def login():
    return render_template('login.html')  
              
@app.route('/dashboard', methods = ['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == '12345':
            return render_template('dashboard.html', u = username, p = password)
        else:
            return redirect( url_for('login'),404)

@app.route('/code')
def code():
    akn_nlp.backend()
    res = pd.read_csv('ans.csv')
    neg = res.iloc[0,1]
    pos = res.iloc[1,1]
    
    return render_template('code.html', n = neg, p = pos)

@app.route('/graph')
def graph():  
    #pie chart  
    return render_template('graph.html')

@app.route('/graph1')
def graph1(): 
    #bar graph   
    return render_template('graph1.html')

if __name__ == "__main__":
    app.run(port=4969, debug=True)