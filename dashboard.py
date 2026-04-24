import os
from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow any frontend to call this API

# ─── MONGODB CONFIG ───────────────────────────────────
MONGO_URI       = os.environ.get('MONGO_URI', 'mongodb+srv://wasteclassifier:waste@cluster0.dfks8gq.mongodb.net/?appName=Cluster0')
DB_NAME         = 'waste_classifier'
COLLECTION_NAME = 'detections'
# ──────────────────────────────────────────────────────

client     = MongoClient(MONGO_URI)
db         = client[DB_NAME]
collection = db[COLLECTION_NAME]

BIN_COLORS = {
    'RECYCLE BIN (Plastic)' : '#3B82F6',
    'RECYCLE BIN (Paper)'   : '#10B981',
    'RECYCLE BIN (Glass)'   : '#8B5CF6',
    'RECYCLE BIN (Metal)'   : '#F59E0B',
    'ORGANIC BIN'           : '#84CC16',
    'GENERAL BIN (Non-recyclable)': '#6B7280',
}


# ─── FRONTEND ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ─── API ENDPOINTS ────────────────────────────────────
@app.route('/api/stats')
def stats():
    total    = collection.count_documents({})
    correct  = collection.count_documents({'is_correct': True})
    accuracy = round((correct / total * 100), 1) if total > 0 else 0

    bin_counts = {}
    for doc in collection.find({}, {'bin_label': 1}):
        bin_label = doc.get('bin_label', 'Unknown')
        bin_counts[bin_label] = bin_counts.get(bin_label, 0) + 1

    label_counts = {}
    for doc in collection.find({}, {'detected_label': 1}):
        label = doc.get('detected_label', 'unknown').capitalize()
        label_counts[label] = label_counts.get(label, 0) + 1

    return jsonify({
        'total'        : total,
        'correct'      : correct,
        'accuracy'     : accuracy,
        'bin_counts'   : bin_counts,
        'label_counts' : label_counts,
        'bin_colors'   : BIN_COLORS,
    })


@app.route('/api/detections')
def detections():
    docs   = list(collection.find().sort('timestamp', -1).limit(20))
    result = []
    for doc in docs:
        ts = doc.get('timestamp')
        result.append({
            'id'             : str(doc['_id']),
            'timestamp'      : ts.strftime('%Y-%m-%d %H:%M:%S') if ts else 'N/A',
            'detected_label' : doc.get('detected_label', 'unknown').capitalize(),
            'confidence'     : round(doc.get('confidence', 0) * 100, 1),
            'waste_type'     : doc.get('waste_type', 'N/A'),
            'bin_label'      : doc.get('bin_label', 'N/A'),
            'is_correct'     : doc.get('is_correct', False),
            'photo1'         : doc.get('photo1_url', doc.get('photo1_path', '')),
            'photo2'         : doc.get('photo2_url', doc.get('photo2_path', '')),
        })
    return jsonify(result)


@app.route('/api/detection', methods=['POST'])
def add_detection():
    """Endpoint for waste_classifier.py to POST new detections."""
    from flask import request
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data'}), 400

    collection.insert_one({
        'timestamp'      : datetime.now(),
        'detected_label' : data.get('detected_label', 'unknown'),
        'confidence'     : data.get('confidence', 0),
        'waste_type'     : data.get('waste_type', 'GENERAL'),
        'bin_label'      : data.get('bin_label', 'GENERAL BIN'),
        'is_correct'     : data.get('is_correct', False),
        'photo1_path'    : data.get('photo1_path', ''),
        'photo2_path'    : data.get('photo2_path', ''),
    })
    return jsonify({'status': 'ok'}), 201


@app.route('/captures/<path:filename>')
def captures(filename):
    captures_dir = os.path.join(os.path.dirname(__file__), 'captures')
    return send_from_directory(captures_dir, filename)


# ─── RUN ──────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)