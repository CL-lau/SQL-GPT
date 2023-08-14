from flask import Flask, render_template, request, jsonify
import openai

app = Flask(__name__)

# 设置您的OpenAI API密钥
openai.api_key = "YOUR_OPENAI_API_KEY"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action = request.form['action']
        description = request.form['description']
        sql_query = request.form['sql_query']
        database_structure = request.form['database_structure']

        if action == 'generate':
            generated_query = generate_sql_query(description)
            return render_template('index.html', generated_query=generated_query)
        elif action == 'optimize':
            optimized_query = optimize_sql_query(sql_query, database_structure)
            return render_template('index.html', optimized_query=optimized_query)

    return render_template('index.html')


def generate_sql_query(description):
    prompt = f"Generate a SQL query to {description}."
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()


def optimize_sql_query(sql_query, database_structure):
    prompt = f"Optimize the following SQL query:\n\n{sql_query}\n\nusing the database structure:\n\n{database_structure}"
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()


if __name__ == '__main__':
    app.run(debug=True)
