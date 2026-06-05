import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import multiprocessing as mp

# Завантаження лексикону VADER
nltk.download('vader_lexicon', quiet=True)


# --- ФУНКЦІЯ ДЛЯ ПАРАЛЕЛЬНИХ ОБЧИСЛЕНЬ ---
# Ця функція працює ізольовано всередині кожного ядра
def analyze_chunk(chunk):
    # Створюємо аналізатор VADER ОДИН РАЗ для всього шматка даних у цьому ядрі
    sia = SentimentIntensityAnalyzer()

    def get_score(text):
        if not isinstance(text, str) or not text.strip():
            return 0.0
        return sia.polarity_scores(text)['compound']

    # Швидко застосовуємо аналіз до всіх рядків у шматку
    return chunk.apply(get_score)


# --- ГОЛОВНИЙ БЛОК ---
if __name__ == '__main__':
    print("1. Завантаження даних...")
    # dtype=str запобігає DtypeWarning
    df_reviews = pd.read_csv("../Books_rating.csv", dtype=str)
    my_books = pd.read_csv("../data/BX-Books.csv", delimiter=';', encoding='ISO-8859-1', on_bad_lines='skip')

    book_title_col = 'Book-Title' if 'Book-Title' in my_books.columns else 'title'
    isbn_col = 'ISBN' if 'ISBN' in my_books.columns else 'isbn'

    my_books['isbn_str'] = my_books[isbn_col].astype(str).str.strip()
    my_books['merge_title'] = my_books[book_title_col].astype(str).str.lower().str.strip()

    df_reviews['Id_str'] = df_reviews['Id'].astype(str).str.strip()
    df_reviews['merge_title'] = df_reviews['Title'].astype(str).str.lower().str.strip()

    print("2. Зіставлення Pass 1 (Id → ISBN)...")
    df_reviews['temp_review_id'] = range(len(df_reviews))

    pass1_matched = pd.merge(
        df_reviews,
        my_books[['isbn_str', isbn_col]],
        left_on='Id_str',
        right_on='isbn_str',
        how='inner'
    )
    pass1_matched = pass1_matched.drop(columns=['isbn_str'])
    pass1_matched.rename(columns={isbn_col: 'final_isbn'}, inplace=True)

    print(f"   ✓ Знайдено через Id: {len(pass1_matched)} відгуків.")

    matched_review_ids = pass1_matched['temp_review_id'].tolist()
    unmatched_reviews = df_reviews[~df_reviews['temp_review_id'].isin(matched_review_ids)]

    print("3. Зіставлення Pass 2 (Title → Нормалізована назва)...")
    unique_books_by_title = my_books.drop_duplicates(subset=['merge_title'])

    pass2_matched = pd.merge(
        unmatched_reviews,
        unique_books_by_title[['merge_title', isbn_col]],
        on='merge_title',
        how='inner'
    )
    pass2_matched.rename(columns={isbn_col: 'final_isbn'}, inplace=True)

    print(f"   ✓ Знайдено через Title: {len(pass2_matched)} відгуків.")

    df_final = pd.concat([pass1_matched, pass2_matched], ignore_index=True)
    df_final = df_final.drop(columns=['temp_review_id', 'Id_str', 'merge_title'])

    print(f"Загалом відібрано {len(df_final)} відгуків для NLP-аналізу.")

    print("4. Підготовка тексту (Summary + Text)...")
    df_final['review/summary'] = df_final['review/summary'].fillna('')
    df_final['review/text'] = df_final['review/text'].fillna('')
    df_final['full_text'] = df_final['review/summary'] + ". " + df_final['review/text']

    df_final['clean_text'] = (
        df_final['full_text']
        .str.replace(r'https?:\/\/.*[\r\n]*', '', regex=True)
        .str.replace(r'@[A-Za-z0-9]+', '', regex=True)
    )

    print("5. Розрахунок Sentiment (VADER) паралельно на всіх ядрах...")

    # Визначаємо кількість ядер вашого процесора
    num_cores = mp.cpu_count()

    # Розрізаємо масив текстів на рівні частини (по одній на кожне ядро)
    chunks = np.array_split(df_final['clean_text'], num_cores)

    # Запускаємо стандартний Pool процесорів (надійний спосіб для Windows)
    with mp.Pool(processes=num_cores) as pool:
        # pool.map роздає шматки ядрам і чекає на результат
        results = pool.map(analyze_chunk, chunks)

    # Зшиваємо результати назад в одну колонку
    df_final['polarity'] = pd.concat(results)

    print("6. Класифікація (Positive/Negative/Neutral)...")
    conditions = [
        (df_final['polarity'] >= 0.05),
        (df_final['polarity'] <= -0.05)
    ]
    choices = ['Positive', 'Negative']
    df_final['analysis'] = np.select(conditions, choices, default='Neutral')

    print("7. Очищення датафрейму та збереження результату...")
    final_columns = ['Id', 'final_isbn', 'Title', 'review/score', 'polarity', 'analysis']

    df_export = df_final[final_columns]
    df_export.to_csv('../sentiments.csv', index=False)

    print("Готово! Файл sentiments.csv успішно згенеровано.")
