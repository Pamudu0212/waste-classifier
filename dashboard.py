import os
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime, timedelta
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ─── MONGODB ──────────────────────────────────────────
MONGO_URI       = os.environ.get('MONGO_URI', 'mongodb+srv://wasteclassifier:waste@cluster0.dfks8gq.mongodb.net/?appName=Cluster0')
client          = MongoClient(MONGO_URI)
db              = client['waste_classifier']
collection      = db['detections']

# ─── OPENROUTER ───────────────────────────────────────
OPENROUTER_KEY  = os.environ.get('OPENROUTER_API_KEY', '')

# ─── CSV DATASET ──────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), 'waste_segregation_sl.csv')
df = pd.read_csv(CSV_PATH)

# ─── BIN REGISTRY ─────────────────────────────────────
BIN_REGISTRY = {
    'BIN_01': {
        'device_id' : 'ESP32_001',
        'bin_type'  : 'plastic',
        'location'  : 'SLIIT Campus - Block A',
        'status'    : 'active',
    },
    'BIN_02': {
        'device_id' : 'ESP32_002',
        'bin_type'  : 'paper',
        'location'  : 'SLIIT Campus - Block B',
        'status'    : 'active',
    },
    'BIN_03': {
        'device_id' : 'ESP32_003',
        'bin_type'  : 'glass',
        'location'  : 'SLIIT Campus - Block C',
        'status'    : 'active',
    },
    'BIN_04': {
        'device_id' : 'ESP32_004',
        'bin_type'  : 'metal',
        'location'  : 'SLIIT Campus - Block D',
        'status'    : 'active',
    },
}

BIN_COLORS = {
    'RECYCLE BIN (Plastic)' : '#3B82F6',
    'RECYCLE BIN (Paper)'   : '#10B981',
    'RECYCLE BIN (Glass)'   : '#8B5CF6',
    'RECYCLE BIN (Metal)'   : '#F59E0B',
    'ORGANIC BIN'           : '#84CC16',
    'GENERAL BIN (Non-recyclable)': '#6B7280',
}


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def index():
    return render_template('index.html')


# ─── EXISTING IOT ENDPOINTS ───────────────────────────
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
            'device_id'      : doc.get('device_id', 'N/A'),
            'bin_id'         : doc.get('bin_id', 'N/A'),
            'location'       : doc.get('location', 'N/A'),
        })
    return jsonify(result)


@app.route('/api/detection', methods=['POST'])
def add_detection():
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
        'device_id'      : data.get('device_id', 'UNKNOWN'),
        'bin_id'         : data.get('bin_id', 'UNKNOWN'),
        'bin_type'       : data.get('bin_type', 'unknown'),
        'location'       : data.get('location', 'N/A'),
        'photo1_path'    : data.get('photo1_path', ''),
        'photo2_path'    : data.get('photo2_path', ''),
    })
    return jsonify({'status': 'ok'}), 201


# ─── BIN ENDPOINTS ────────────────────────────────────
@app.route('/api/bins')
def bins():
    """All bins with their stats."""
    result = []
    for bin_id, info in BIN_REGISTRY.items():
        total   = collection.count_documents({'bin_id': bin_id})
        correct = collection.count_documents({'bin_id': bin_id, 'is_correct': True})
        wrong   = total - correct
        accuracy = round(correct / total * 100, 1) if total > 0 else 0

        # Most recent detection
        latest = collection.find_one({'bin_id': bin_id}, sort=[('timestamp', -1)])
        last_seen = 'Never'
        if latest and latest.get('timestamp'):
            last_seen = latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

        # Wrong disposal types
        wrong_types = list(collection.aggregate([
            {'$match': {'bin_id': bin_id, 'is_correct': False}},
            {'$group': {'_id': '$detected_label', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}, {'$limit': 3}
        ]))

        result.append({
            'bin_id'      : bin_id,
            'device_id'   : info['device_id'],
            'bin_type'    : info['bin_type'],
            'location'    : info['location'],
            'status'      : info['status'],
            'total'       : total,
            'correct'     : correct,
            'wrong'       : wrong,
            'accuracy'    : accuracy,
            'last_seen'   : last_seen,
            'wrong_types' : [{'label': w['_id'], 'count': w['count']} for w in wrong_types],
        })
    return jsonify(result)


@app.route('/api/bins/<bin_id>')
def bin_detail(bin_id):
    """Detailed analytics for a specific bin."""
    if bin_id not in BIN_REGISTRY:
        return jsonify({'error': 'Bin not found'}), 404

    info    = BIN_REGISTRY[bin_id]
    total   = collection.count_documents({'bin_id': bin_id})
    correct = collection.count_documents({'bin_id': bin_id, 'is_correct': True})
    wrong   = total - correct
    accuracy = round(correct / total * 100, 1) if total > 0 else 0

    # Wrong disposal breakdown
    wrong_by_type = list(collection.aggregate([
        {'$match': {'bin_id': bin_id, 'is_correct': False}},
        {'$group': {'_id': '$detected_label', 'count': {'$sum': 1}}},
        {'$sort': {'count': -1}}
    ]))

    # Recent events
    recent = list(collection.find({'bin_id': bin_id}).sort('timestamp', -1).limit(10))
    recent_events = []
    for doc in recent:
        ts = doc.get('timestamp')
        recent_events.append({
            'timestamp'      : ts.strftime('%H:%M:%S') if ts else 'N/A',
            'detected_label' : doc.get('detected_label', 'unknown').capitalize(),
            'confidence'     : round(doc.get('confidence', 0) * 100, 1),
            'is_correct'     : doc.get('is_correct', False),
        })

    # Daily trend (last 7 days)
    daily = []
    for i in range(6, -1, -1):
        day_start = datetime.utcnow().replace(hour=0, minute=0, second=0) - timedelta(days=i)
        day_end   = day_start + timedelta(days=1)
        count     = collection.count_documents({'bin_id': bin_id, 'timestamp': {'$gte': day_start, '$lt': day_end}})
        daily.append({'date': day_start.strftime('%a'), 'count': count})

    return jsonify({
        'bin_id'         : bin_id,
        'device_id'      : info['device_id'],
        'bin_type'       : info['bin_type'],
        'location'       : info['location'],
        'status'         : info['status'],
        'total'          : total,
        'correct'        : correct,
        'wrong'          : wrong,
        'accuracy'       : accuracy,
        'wrong_by_type'  : [{'label': w['_id'], 'count': w['count']} for w in wrong_by_type],
        'recent_events'  : recent_events,
        'daily_trend'    : daily,
    })


# ─── DEVICE ENDPOINTS ─────────────────────────────────
@app.route('/api/devices')
def devices():
    """All devices with real-time health status from heartbeat collection."""
    heartbeats = db['heartbeats']
    result = []

    for bin_id, info in BIN_REGISTRY.items():
        device_id = info['device_id']
        total     = collection.count_documents({'device_id': device_id})

        # Check heartbeat first (real-time)
        hb = heartbeats.find_one({'device_id': device_id})

        last_seen = None
        status    = 'offline'

        if hb and hb.get('timestamp'):
            last_seen   = hb['timestamp']
            seconds_ago = (datetime.utcnow() - last_seen).total_seconds()
            if seconds_ago < 45:      # heartbeat within 45s = online
                status = 'online'
            elif seconds_ago < 90:    # within 90s = idle (transitioning)
                status = 'idle'
            else:
                status = 'offline'   # no heartbeat for 90s+ = offline
        else:
            # Fallback: check last detection
            latest = collection.find_one({'device_id': device_id}, sort=[('timestamp', -1)])
            if latest and latest.get('timestamp'):
                last_seen   = latest['timestamp']
                seconds_ago = (datetime.utcnow() - last_seen).total_seconds()
                if seconds_ago < 3600:
                    status = 'idle'
                else:
                    status = 'offline'

        if total == 0 and not hb:
            status = 'offline'

        result.append({
            'device_id'   : device_id,
            'bin_id'      : bin_id,
            'bin_type'    : info['bin_type'],
            'location'    : info['location'],
            'status'      : status,
            'total_events': total,
            'last_seen'   : last_seen.strftime('%Y-%m-%d %H:%M:%S') if last_seen else 'Never',
        })
    return jsonify(result)


# ─── CSV ENDPOINTS ────────────────────────────────────
@app.route('/api/csv/stats')
def csv_stats():
    return jsonify({
        'total_rows'         : len(df),
        'years'              : sorted(df['year'].unique().tolist()),
        'districts'          : sorted(df['district'].unique().tolist()),
        'material_types'     : sorted(df['material_type'].unique().tolist()),
        'avg_wrong_disposal' : round(df['wrong_disposal_rate_percent'].mean(), 2),
        'avg_recycling_rate' : round(df['recycling_rate_percent'].mean(), 2),
        'avg_segregation'    : round(df['correctly_segregated_percent'].mean(), 2),
    })


@app.route('/api/csv/by-material')
def csv_by_material():
    grouped = df.groupby('material_type').agg({
        'wrong_disposal_rate_percent'  : 'mean',
        'correctly_segregated_percent' : 'mean',
        'recycling_rate_percent'       : 'mean',
        'total_generated_tons'         : 'sum',
    }).round(2).reset_index()
    return jsonify(grouped.to_dict(orient='records'))


@app.route('/api/csv/trends')
def csv_trends():
    grouped = df.groupby('year').agg({
        'wrong_disposal_rate_percent'  : 'mean',
        'correctly_segregated_percent' : 'mean',
        'recycling_rate_percent'       : 'mean',
    }).round(2).reset_index()
    return jsonify(grouped.to_dict(orient='records'))


@app.route('/api/csv/urban-rural')
def csv_urban_rural():
    grouped = df.groupby('urban_rural').agg({
        'wrong_disposal_rate_percent'  : 'mean',
        'correctly_segregated_percent' : 'mean',
        'recycling_rate_percent'       : 'mean',
    }).round(2).reset_index()
    return jsonify(grouped.to_dict(orient='records'))


@app.route('/captures/<path:filename>')
def captures(filename):
    captures_dir = os.path.join(os.path.dirname(__file__), 'captures')
    return send_from_directory(captures_dir, filename)


# ─── CONTEXT BUILDERS ─────────────────────────────────
def build_iot_context():
    from datetime import datetime, timedelta

    total   = collection.count_documents({})
    correct = collection.count_documents({'is_correct': True})
    wrong   = total - correct
    accuracy = round(correct / total * 100, 1) if total else 0

    # Top waste types
    labels = list(collection.aggregate([
        {"$group": {"_id": "$detected_label", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}, {"$limit": 5}
    ]))
    label_info = ", ".join([f"{l['_id'].capitalize()}({l['count']})" for l in labels])

    # Wrong disposal rate by waste type
    wrong_by_type = list(collection.aggregate([
        {"$group": {
            "_id": "$detected_label",
            "total": {"$sum": 1},
            "wrong": {"$sum": {"$cond": ["$is_correct", 0, 1]}}
        }},
        {"$project": {
            "wrong_rate": {"$round": [{"$multiply": [{"$divide": ["$wrong", "$total"]}, 100]}, 1]}
        }},
        {"$sort": {"wrong_rate": -1}}, {"$limit": 4}
    ]))
    wrong_type_info = ", ".join([f"{w['_id'].capitalize()}({w['wrong_rate']}%)" for w in wrong_by_type])

    # Bin stats
    bin_stats = []
    for bin_id, info in BIN_REGISTRY.items():
        b_total   = collection.count_documents({'bin_id': bin_id})
        b_correct = collection.count_documents({'bin_id': bin_id, 'is_correct': True})
        b_wrong   = b_total - b_correct
        b_acc     = round(b_correct / b_total * 100, 1) if b_total > 0 else 0
        bin_stats.append(f"{bin_id}({info['bin_type'].upper()}, {b_acc}% correct, {b_wrong} wrong)")
    bin_info = ", ".join(bin_stats)

    # Last wrong disposal
    last_wrong = collection.find_one({"is_correct": False}, sort=[("timestamp", -1)])
    last_wrong_info = "None recorded"
    if last_wrong:
        ts = last_wrong.get('timestamp', '')
        if hasattr(ts, 'strftime'):
            ts = ts.strftime('%Y-%m-%d %H:%M:%S')
        last_wrong_info = f"{last_wrong.get('detected_label','unknown').capitalize()} in {last_wrong.get('bin_id','N/A')} at {ts}"

    # Confidence stats
    conf_stats = list(collection.aggregate([
        {"$group": {
            "_id": None,
            "avg_conf": {"$avg": "$confidence"},
            "low_conf": {"$sum": {"$cond": [{"$lt": ["$confidence", 0.6]}, 1, 0]}}
        }}
    ]))
    avg_conf = round(conf_stats[0]['avg_conf'] * 100, 1) if conf_stats else 0
    low_conf = conf_stats[0]['low_conf'] if conf_stats else 0

    # Today vs yesterday
    now = datetime.utcnow()
    today_start     = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    today_count     = collection.count_documents({"timestamp": {"$gte": today_start}})
    yesterday_count = collection.count_documents({"timestamp": {"$gte": yesterday_start, "$lt": today_start}})

    # Last 5 detections
    recent = list(collection.find().sort('timestamp', -1).limit(5))
    recent_info = ""
    for i, doc in enumerate(recent):
        ts = doc.get('timestamp', '')
        if hasattr(ts, 'strftime'):
            ts = ts.strftime('%H:%M:%S')
        status = "CORRECT" if doc.get('is_correct') else "WRONG"
        recent_info += f"\n  {i+1}. {doc.get('detected_label','unknown').capitalize()} | {round(doc.get('confidence',0)*100,1)}% | {doc.get('bin_id','N/A')} | {status} | {ts}"

    co2_saved = round(correct * 0.5, 1)

    return f"""
LIVE IOT DATA (Real-time from ESP32-CAM system):

SYSTEM OVERVIEW:
- Total detections: {total}
- Correct disposals: {correct} ({accuracy}%)
- Wrong disposals: {wrong} ({round(wrong/total*100,1) if total else 0}%)
- Average confidence: {avg_conf}%
- Low confidence detections: {low_conf}
- Estimated CO2 saved: {co2_saved} kg
- Detections today: {today_count} | Yesterday: {yesterday_count}

BIN PERFORMANCE:
{bin_info}
- Last wrong disposal: {last_wrong_info}

WASTE TYPE ANALYSIS:
- Top detected: {label_info}
- Wrong rate by type: {wrong_type_info}

LAST 5 DETECTIONS:{recent_info}
"""


def build_bin_context():
    """Build detailed per-bin context."""
    bin_contexts = []
    for bin_id, info in BIN_REGISTRY.items():
        total   = collection.count_documents({'bin_id': bin_id})
        correct = collection.count_documents({'bin_id': bin_id, 'is_correct': True})
        wrong   = total - correct
        acc     = round(correct / total * 100, 1) if total > 0 else 0

        wrong_types = list(collection.aggregate([
            {'$match': {'bin_id': bin_id, 'is_correct': False}},
            {'$group': {'_id': '$detected_label', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}, {'$limit': 3}
        ]))
        wrong_type_str = ", ".join([f"{w['_id']}({w['count']})" for w in wrong_types]) or "None"

        bin_contexts.append(f"""
  {bin_id}:
  - Device: {info['device_id']}
  - Type: {info['bin_type'].upper()} bin
  - Location: {info['location']}
  - Total events: {total}
  - Correct: {correct} | Wrong: {wrong} | Accuracy: {acc}%
  - Most wrongly disposed into this bin: {wrong_type_str}
  - Status: {info['status']}""")

    return "BIN REGISTRY:\n" + "\n".join(bin_contexts)


def build_csv_context():
    avg_wrong   = round(df['wrong_disposal_rate_percent'].mean(), 1)
    avg_correct = round(df['correctly_segregated_percent'].mean(), 1)
    avg_recycle = round(df['recycling_rate_percent'].mean(), 1)
    total_co2   = round(df['co2_saved_tons'].sum(), 1)
    total_value = round(df['economic_value_LKR'].sum() / 1e6, 1)

    mat_wrong      = df.groupby('material_type')['wrong_disposal_rate_percent'].mean()
    worst_material = mat_wrong.idxmax()
    best_material  = mat_wrong.idxmin()

    dist_wrong     = df.groupby('district')['wrong_disposal_rate_percent'].mean()
    worst_district = dist_wrong.idxmax()
    best_district  = dist_wrong.idxmin()

    urban = round(df[df['urban_rural'] == 'Urban']['correctly_segregated_percent'].mean(), 1)
    rural = round(df[df['urban_rural'] == 'Rural']['correctly_segregated_percent'].mean(), 1)

    yr2015    = round(df[df['year'] == 2015]['correctly_segregated_percent'].mean(), 1)
    yr2020    = round(df[df['year'] == 2020]['correctly_segregated_percent'].mean(), 1)
    trend     = round(yr2020 - yr2015, 1)
    trend_dir = "improved" if trend > 0 else "declined"

    return f"""
SECONDARY DATASET (Sri Lanka Waste Segregation 2015-2020):
- Avg wrong disposal: {avg_wrong}% | Avg correct: {avg_correct}% | Avg recycling: {avg_recycle}%
- Total CO2 saved nationally: {total_co2} tons | Economic value: LKR {total_value}M
- Worst material: {worst_material} | Best material: {best_material}
- Worst district: {worst_district} | Best district: {best_district}
- Urban: {urban}% correct | Rural: {rural}% correct
- Trend 2015-2020: {trend_dir} by {abs(trend)}% ({yr2015}% -> {yr2020}%)
"""


# ─── ROLE-SPECIFIC SYSTEM PROMPTS ─────────────────────
ROLE_PROMPTS = {
    'supervisor': """You are WasteBot for a Waste Supervisor.
Focus on: KPIs, trends, wrong disposal patterns, bin performance, district comparisons.
Suggest operational improvements. Be analytical and strategic.
Answer questions like: "Which bin has most wrong disposals?", "How are we performing vs national average?"
Keep answers concise but data-driven (3-5 sentences).""",

    'collector': """You are WasteBot for a Garbage Collector.
Focus on: Simple, actionable insights only. Which bins need attention. Recent wrong disposals.
Use simple language. No complex analytics. Be direct and operational.
Answer questions like: "Which bin needs attention?", "What was just disposed wrongly?"
Keep answers very short and action-focused (1-3 sentences).""",

    'admin': """You are WasteBot for a System Administrator/Analyst.
Focus on: Full analytics, device health, bin performance, multi-dimensional insights.
Provide detailed comparisons between IoT live data and Sri Lanka national dataset.
Answer complex questions like device status, accuracy trends, environmental impact.
Be comprehensive and technical (4-6 sentences with specific numbers).""",

    'default': """You are WasteBot for the WasteVision smart waste management system.
Use both live IoT data and Sri Lanka historical dataset to answer questions.
Be concise and data-driven."""
}


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        body     = request.get_json()
        messages = body.get('messages', [])
        role     = body.get('role', 'default')

        # Build rich context
        iot_context = build_iot_context()
        bin_context = build_bin_context()
        csv_context = build_csv_context()

        role_prompt  = ROLE_PROMPTS.get(role, ROLE_PROMPTS['default'])
        full_context = role_prompt + "\n\nDATA CONTEXT:\n" + iot_context + "\n" + bin_context + "\n" + csv_context

        llm = OpenAI(
            base_url = "https://openrouter.ai/api/v1",
            api_key  = OPENROUTER_KEY
        )

        response = llm.chat.completions.create(
            model    = "openrouter/auto",
            messages = [
                {"role": "system", "content": full_context},
                *messages
            ],
            max_tokens = 300
        )

        return jsonify({'reply': response.choices[0].message.content})

    except Exception as e:
        try:
            total    = collection.count_documents({})
            accuracy = round(collection.count_documents({'is_correct': True}) / total * 100, 1) if total else 0
            return jsonify({'reply': f"System has processed {total} detections with {accuracy}% accuracy. AI temporarily unavailable."})
        except:
            return jsonify({'reply': "WasteBot temporarily unavailable."}), 500


# ─── HEALTH CHECK ─────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({
        'status'   : 'ok',
        'mongodb'  : 'connected',
        'csv_rows' : len(df),
        'bins'     : len(BIN_REGISTRY)
    })


# ─── RUN ──────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)