from flask import Flask, render_template_string

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>VAE-Playground</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .badge { background: #28a745; color: white; padding: 5px 15px; border-radius: 15px; }
    </style>
</head>
<body>
    <h1>🧬 VAE-Playground</h1>
    <p>Docker app running on port 7860</p>
    <div class="badge">✓ Running</div>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
