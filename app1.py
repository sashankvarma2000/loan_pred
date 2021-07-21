from flask import Flask,render_template,request
import numpy as np
import pickle



app=Flask(__name__)
Rf=pickle.load(open('loan_pred_model.pkl','rb'))


@app.route("/",methods=["POST","GET"])
def hello():
    loan_pred=None
    if request.method=="POST":
        loan_amt=float(request.form['loan_amt'])
        annual_inc=float(request.form['annual_inc'])
        int_rate=float(request.form['int_rate'])
        tot_cur_bal=float(request.form['tot_cur_bal'])
        arr=np.array([[loan_amt,annual_inc,int_rate,tot_cur_bal]])
        loan_pred=Rf.predict(arr)
        

    return render_template("index.html",my_loan=loan_pred)

if __name__=="__main__":
    app.run(debug=True)