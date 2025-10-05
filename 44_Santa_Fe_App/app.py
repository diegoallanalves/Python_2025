from flask import Flask, render_template, redirect, request, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import timedelta
import os

# Flask application setup
app = Flask(__name__)
app.secret_key = "Your_Secret_Key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///C:\\Users\\diego\\Desktop\\python\\44_Santa_Fe_App\\santa.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

print("Database is saved at: C:\\Users\\diego\\Desktop\\python\\44_Santa_Fe_App\\santa.db")


# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    has_submitted = db.Column(db.Boolean, default=False)
    answers = db.relationship('Answer', backref='user', lazy=True)


class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    response = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())


# Ensure database tables are created
with app.app_context():
    db.create_all()


# Redirect root URL ("/") to appropriate page
@app.route('/')
def home():
    # Check if user is logged in via the session
    if 'user_id' in session:
        user = User.query.get(session['user_id'])  # Retrieve user data
        if user and user.has_submitted:  # If the user has already submitted
            flash("You have already submitted the form!", "info")
            return redirect(url_for('submitted'))
        return redirect(url_for('form'))  # Redirect to form if not submitted yet

    # If not logged in, redirect to login or registration
    return redirect(url_for('login'))


# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        phone = request.form['phone']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        if User.query.filter_by(phone=phone).first():
            flash("Phone number already registered. Please log in.", "danger")
            return redirect(url_for('login'))

        new_user = User(phone=phone, password=password, has_submitted=False)
        db.session.add(new_user)
        db.session.commit()

        flash("Registered successfully. Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        phone = request.form['phone']
        password = request.form['password']

        user = User.query.filter_by(phone=phone).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['phone'] = user.phone
            flash("Login successful!", "success")
            return redirect(url_for('form'))

        flash("Invalid phone number or password.", "danger")
        return redirect(url_for('login'))

    return render_template('login.html')


# Form Submission Route
@app.route('/form', methods=['GET', 'POST'])
def form():
    if 'user_id' not in session:
        flash("Please log in to access the form.", "danger")
        return redirect(url_for('login'))

    # Fetch the currently logged-in user
    user = User.query.get(session['user_id'])

    # Handle the case where the user is not found in the database
    if user is None:
        session.clear()  # Clear the session to prevent invalid IDs
        flash("Session is invalid. Please log in again.", "danger")
        return redirect(url_for('login'))

    # Check if the user has already submitted the form
    if user.has_submitted:
        flash("You have already submitted the form. Thank you!", "info")
        return redirect(url_for('submitted'))

    if request.method == 'POST':
        response = request.form['response']

        # Save the form response in the database
        new_answer = Answer(user_id=user.id, response=response)
        db.session.add(new_answer)

        # Mark user as having submitted
        user.has_submitted = True
        db.session.commit()

        flash("Form submitted successfully!", "success")
        return redirect(url_for('submitted'))

    return render_template('form.html')


# Thank You Route
@app.route('/submitted')
def submitted():
    if 'user_id' not in session:
        flash("Please log in first.", "danger")
        return redirect(url_for('login'))

    return render_template('submitted.html')


# Logout Route
@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))


# View All Submitted Answers
@app.route('/answers')
def view_answers():
    answers = Answer.query.all()
    return render_template('answers.html', answers=answers)


if __name__ == '__main__':
    app.run(debug=True)
