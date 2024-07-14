from flask import Flask, render_template
import pronosticos
import stats

app = Flask(__name__)
app.register_blueprint(pronosticos.bp)
app.register_blueprint(stats.bp)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
 