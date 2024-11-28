from flask import Flask, render_template, abort
from datetime import datetime
from sqlalchemy import distinct, func
from config import SessionLocal
from models.db_models import GeneratedNews, Section

app = Flask(__name__)


def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


@app.template_filter('format_date')
def format_date(date):
    """Convert date to Spanish format"""
    return date.strftime("%d de %B de %Y").replace("January", "Enero") \
        .replace("February", "Febrero").replace("March", "Marzo") \
        .replace("April", "Abril").replace("May", "Mayo") \
        .replace("June", "Junio").replace("July", "Julio") \
        .replace("August", "Agosto").replace("September", "Septiembre") \
        .replace("October", "Octubre").replace("November", "Noviembre") \
        .replace("December", "Diciembre")


@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}


@app.route('/')
def home():
    db = get_db()
    # Get today's date (without time)
    today = datetime.utcnow().date()

    # Get all sections with their news for today
    sections = db.query(Section).all()

    # Get all news for today grouped by section
    news_by_section = {}
    for section in sections:
        news = db.query(GeneratedNews) \
            .filter(func.date(GeneratedNews.generated_at) == today) \
            .filter(GeneratedNews.section_id == section.id) \
            .all()
        news_by_section[section.name] = news

    # Get all available dates for navigation
    available_dates = db.query(
        func.date(GeneratedNews.generated_at)
    ).distinct().order_by(
        func.date(GeneratedNews.generated_at).desc()
    ).all()
    available_dates = [date[0] for date in available_dates]

    return render_template('index.html',
                           news_by_section=news_by_section,
                           current_date=today,
                           available_dates=available_dates)


@app.route('/noticias/<string:date>')
def news_by_date(date):
    try:
        # Convert string date to datetime
        selected_date = datetime.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        abort(404)

    db = get_db()
    sections = db.query(Section).all()

    # Get news for selected date grouped by section
    news_by_section = {}
    for section in sections:
        news = db.query(GeneratedNews) \
            .filter(func.date(GeneratedNews.generated_at) == selected_date) \
            .filter(GeneratedNews.section_id == section.id) \
            .all()
        news_by_section[section.name] = news

    # Get all available dates for navigation
    available_dates = db.query(
        func.date(GeneratedNews.generated_at)
    ).distinct().order_by(
        func.date(GeneratedNews.generated_at).desc()
    ).all()
    available_dates = [date[0] for date in available_dates]

    return render_template('news_by_date.html',
                           news_by_section=news_by_section,
                           current_date=selected_date,
                           available_dates=available_dates)


@app.route('/noticia/<int:news_id>')
def article(news_id):
    db = get_db()
    news = db.query(GeneratedNews).get(news_id)
    if news is None:
        abort(404)

    return render_template('article.html', article=news)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)
