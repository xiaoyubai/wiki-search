from collections import Counter
from flask import Flask, request, redirect, render_template
app = Flask(__name__)
import cPickle as pickle
import pandas as pd



def dict_to_html(d):
    return '<br>'.join('{0}: {1}'.format(k, d[k]) for k in sorted(d))


# Form page to submit text
@app.route('/')
def submission_page():
    return render_template("index.html")
    # return '''
    #     <center><img src="https://www.wikipedia.org/portal/wikipedia.org/assets/img/Wikipedia_wordmark_2x.png"
    #     srcset="https://www.wikipedia.org/portal/wikipedia.org/assets/img/Wikipedia_wordmark_2x.png, https://www.wikipedia.org/portal/wikipedia.org/assets/img/Wikipedia_wordmark_2x.png"
    #     width="474" height="80" alt="WikipediA" title="Wikipedia"><center>
    #
    #     <center><form action="/submit" method='POST'>
    #         <input type="text" name="user_input" placeholder="Search ..." />
    #         <input type="submit" />
    #     </form></center>
    #     '''


# My word counter app
@app.route('/submit', methods=['POST'] )
def word_counter():
    text = str(request.form['user_input'])
    keyword, category = text.split(" in ")

    search_words = "+".join(text.split())
    link = "http://en.wikipedia.org/wiki/Special:Search?search=%s&go=Go" % search_words
    return redirect(link)
    # x = text.split(' ')
    #
    # with open('data/vectorizer.pkl') as f:
    #     vectorizer = pickle.load(f)
    # with open('data/model.pkl') as f:
    #     model = pickle.load(f)
    #
    # x = vectorizer.transform(x)
    # pred = model.predict(x)
    #
    # page = 'Predicted section is: {0}.<br>'
    # return page.format(pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
