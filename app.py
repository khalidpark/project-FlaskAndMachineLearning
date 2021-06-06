from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import joblib
import pandas as pd

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///posts.db'
db = SQLAlchemy(app)

# 명령어
# from app import db
# db.create_all()
# from app import BlogPost

class BlogPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)

    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return 'Blog post ' + str(self.id)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/posts', methods=['GET', 'POST'])
def posts():

    if request.method == 'POST':
        post_title = request.form['title']
        post_content = request.form['content']
        new_post = BlogPost(title=post_title, content=post_content)
        db.session.add(new_post)
        db.session.commit()
        return redirect('/posts')
    else:
        all_posts = BlogPost.query.order_by(BlogPost.date_posted).all()
        return render_template('posts.html', posts=all_posts)


@app.route('/posts/delete/<int:id>')
def delete(id):
    post = BlogPost.query.get_or_404(id)
    db.session.delete(post)
    db.session.commit()
    return redirect('/posts')


@app.route('/posts/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    
    post = BlogPost.query.get_or_404(id)

    if request.method == 'POST':
        post.title = request.form['title']
        post.content = request.form['content']
        db.session.commit()
        return redirect('/posts')
    else:
        return render_template('edit.html', post=post)


@app.route('/posts/new', methods=['GET', 'POST'])
def new_post():
    if request.method == 'POST':
        post.title = request.form['title']
        post.content = request.form['content']
        new_post = BlogPost(title=post_title, content=post_content)
        db.session.add(new_post)
        db.session.commit()
        return redirect('/posts')
    else:
        return render_template('new_post.html')


@app.route('/model')
def model():
    return render_template('model.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method =='POST':

        n1 = request.form['number1']
        n2 = request.form['number2']
        n3 = request.form['number3']

        data = {'요일': [n1],'본사정원수': [n2],'본사출장자수': [n3]}
        df = pd.DataFrame(data)
     
        predict = model.predict(df)

        return render_template('model.html',predict=predict)

if __name__ == "__main__":
    model = joblib.load('./model/model.pkl')
    app.run(debug=True)