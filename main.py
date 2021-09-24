from flask.templating import render_template
import joblib
import pandas as pd
from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

def predict():

    data = pd.DataFrame([[0.04741, 	0.0 ,	11.93 ,	0.0 ,	0.573 ,	6.030 ,	80.8 ,	2.5050 ,	1.0 ,	273.0 ,	21.0 	,396.90 ,	7.88 ]], columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
        'PTRATIO', 'B', 'LSTAT'])


    lin_reg = joblib.load('Linear-Regression-Model.pkl')
    numerical_transformer = joblib.load('numerical_transformer.pkl')
    column_transformer= joblib.load('column_transformer.pkl')


    prepared_data = column_transformer.transform(data)
    output = lin_reg.predict(prepared_data)
    final_output = numerical_transformer.inverse_transform(output)

    return '{:.2f}'.format(final_output[0, 0])

print(predict())

if __name__ == '__main__':
    app.run(debug=True)
