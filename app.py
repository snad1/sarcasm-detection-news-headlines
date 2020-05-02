from flask import Flask, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField, TextField, TextAreaField
import sarcasm_model
import numpy
import pytesseract
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment = SentimentIntensityAnalyzer()

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba24qbhgchkhtftde'

model = sarcasm_model.test_model()
model.load_trained_model()


class HeadlineForm(FlaskForm):
    # headline = StringField('Headline')
    headline = TextAreaField('Headline')
    # headline = StringField('Headline', validators=[DataRequired(), Length(min=2, max=200)])
    # content = TextAreaField('Comment')
    image = FileField('Headline image', validators=[FileAllowed(['jpg', 'jpeg', 'png'])])
    submit = SubmitField('Check')


@app.route('/h')
def hello_world():
    return 'Hello World!'


@app.route("/", methods=['GET', 'POST'])
def home():
    headlineForm = HeadlineForm()
    label = []
    results = []
    predicted = -1
    errMsg = ""
    if headlineForm.validate_on_submit():
        # print(headlineForm.image.data.filename)
        if headlineForm.image.data is not None:
            nn = pytesseract.image_to_string(Image.open(headlineForm.image.data))
            headlineForm.headline.data = nn
            # print(nn)
        else:
            nn = headlineForm.headline.data

        if nn != "":
            try:
                # print(nn)
                label = model.predict(nn)
                print(label)
                predicted = numpy.argmax(label)
                label[0] = round(label[0] * 100, 2)
                label[1] = round(label[1] * 100, 2)
                results = sentiment.polarity_scores(headlineForm.headline.data)
                results['compound'] = round(results['compound'] * 100, 2)
                results['pos'] = round(results['pos'] * 100, 2)
                results['neu'] = round(results['neu'] * 100, 2)
                results['neg'] = round(results['neg'] * 100, 2)
                # print('results----------------', results)
            except Exception as e:
                print('Error-ttt:', e)
                # raise

        else:
            headlineForm.headline.errors = ['Empty text']
            # print(headlineForm.headline.errors)

    # return render_template('index.html')
    # print('home')
    return render_template('index.html', headlineForm=headlineForm, label=label, predicted=predicted, results=results)


if __name__ == '__main__':
    app.run()
