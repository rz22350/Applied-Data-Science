from flask import Flask, send_file, request, abort
import os

app = Flask(__name__)

@app.route('/get_image')
def get_image():
    query_path = request.args.get('query_path')
    if not query_path:
        abort(400, description="Missing query_path parameter")
    if not os.path.exists(query_path):
        abort(404, description="Image not found")
    
    valid_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    ext = os.path.splitext(query_path)[1].lower()
    if ext not in valid_extensions:
        abort(400, description="Unsupported file format")
    
    if ext in ['.jpg', '.jpeg']:
        mimetype = 'image/jpeg'
    elif ext == '.png':
        mimetype = 'image/png'
    elif ext == '.gif':
        mimetype = 'image/gif'
    elif ext == '.bmp':
        mimetype = 'image/bmp'
    else:
        mimetype = 'application/octet-stream'
    
    return send_file(query_path, mimetype=mimetype)

if __name__ == '__main__':
    app.run(debug=True)
