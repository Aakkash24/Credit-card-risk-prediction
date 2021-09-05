from flask import Flask, render_template, request
import creditcard

app = Flask(__name__)

"""
@app.route("/sub", methods=["POST"])
def submit():
    if request.method == "POST":
        name = request.form["username"]
    return render_template("sub.html", n=name)
"""


@app.route("/", methods=['POST', 'GET'])
def hello():
    global res
    res = None
    if request.method == "POST":
        Age = request.form['Age']
        Job = request.form['Job']
        ca = request.form['ca']
        duration = request.form['duration']
        sex_f = request.form['sex_f']
        sex_m = request.form['sex_m']
        hf = request.form['hf']
        ho = request.form['ho']
        hr = request.form['hr']
        sal = request.form['sal']
        sam = request.form['sam']
        saq = request.form['saq']
        sar = request.form['sar']
        cal = request.form['cal']
        cam = request.form['cam']
        car = request.form['car']
        pb = request.form['pb']
        pc = request.form['pc']
        pd = request.form['pd']
        pe = request.form['pe']
        pf = request.form['pf']
        pr = request.form['pr']
        pre = request.form['pre']
        pv = request.form['pv']
        res = creditcard.credit_card_risk_prediction(
            abc=[Age, Job, ca, duration, sex_f, sex_m, hf, ho, hr, sal, sam, saq, sar, cal, cam, car, pb, pc, pd, pe, pf, pr, pre, pv])
    return render_template("index.html", pred=res)


if __name__ == '__main__':
    app.run(debug=True)
