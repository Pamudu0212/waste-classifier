import os
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ─── MONGODB ──────────────────────────────────────────
MONGO_URI       = os.environ.get('MONGO_URI', 'mongodb+srv://wasteclassifier:waste@cluster0.dfks8gq.mongodb.net/?appName=Cluster0')
client          = MongoClient(MONGO_URI)
db              = client['waste_classifier']
collection      = db['detections']

# ─── OPENROUTER (Chatbot) ─────────────────────────────
OPENROUTER_KEY  = os.environ.get('OPENROUTER_API_KEY', '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')

# ─── CSV DATASET ──────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), 'waste_segregation_sl.csv')
df = pd.read_csv(CSV_PATH)

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


# ─── EXISTING IOT API ENDPOINTS ───────────────────────
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


# ─── CSV DATASET API ENDPOINTS ────────────────────────
@app.route('/api/csv/stats')
def csv_stats():
    return jsonify({
        'total_rows'      : len(df),
        'years'           : sorted(df['year'].unique().tolist()),
        'districts'       : sorted(df['district'].unique().tolist()),
        'material_types'  : sorted(df['material_type'].unique().tolist()),
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


@app.route('/api/csv/by-district')
def csv_by_district():
    grouped = df.groupby('district').agg({
        'wrong_disposal_rate_percent'  : 'mean',
        'correctly_segregated_percent' : 'mean',
        'recycling_rate_percent'       : 'mean',
    }).round(2).reset_index()
    return jsonify(grouped.to_dict(orient='records'))


@app.route('/api/csv/trends')
def csv_trends():
    grouped = df.groupby('year').agg({
        'wrong_disposal_rate_percent'  : 'mean',
        'correctly_segregated_percent' : 'mean',
        'recycling_rate_percent'       : 'mean',
        'total_generated_tons'         : 'sum',
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


# ─── CHATBOT ──────────────────────────────────────────
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

    # Worst bins
    wrong_bins = list(collection.aggregate([
        {"$match": {"is_correct": False}},
        {"$group": {"_id": "$bin_label", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}, {"$limit": 3}
    ]))
    wrong_bin_info = ", ".join([f"{b['_id']}({b['count']})" for b in wrong_bins])

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

    # Hardest waste type to classify
    conf_by_type = list(collection.aggregate([
        {"$group": {"_id": "$detected_label", "avg_conf": {"$avg": "$confidence"}}},
        {"$sort": {"avg_conf": 1}}, {"$limit": 1}
    ]))
    hardest = conf_by_type[0]['_id'].capitalize() if conf_by_type else "N/A"

    # Today vs yesterday
    now = datetime.utcnow()
    today_start     = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    today_count     = collection.count_documents({"timestamp": {"$gte": today_start}})
    yesterday_count = collection.count_documents({"timestamp": {"$gte": yesterday_start, "$lt": today_start}})

    # Peak hour
    hourly = list(collection.aggregate([
        {"$group": {"_id": {"$hour": "$timestamp"}, "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}, {"$limit": 1}
    ]))
    peak_hour = f"{hourly[0]['_id']}:00" if hourly else "N/A"

    # Last wrong disposal
    last_wrong = collection.find_one({"is_correct": False}, sort=[("timestamp", -1)])
    last_wrong_info = "None recorded"
    if last_wrong:
        ts = last_wrong.get('timestamp', '')
        if hasattr(ts, 'strftime'):
            ts = ts.strftime('%Y-%m-%d %H:%M:%S')
        last_wrong_info = f"{last_wrong.get('detected_label','unknown').capitalize()} in {last_wrong.get('bin_label','N/A')} at {ts}"

    # Last 5 detections
    recent = list(collection.find().sort('timestamp', -1).limit(5))
    recent_info = ""
    for i, doc in enumerate(recent):
        ts = doc.get('timestamp', '')
        if hasattr(ts, 'strftime'):
            ts = ts.strftime('%H:%M:%S')
        status = "CORRECT" if doc.get('is_correct') else "WRONG"
        recent_info += f"\n  {i+1}. {doc.get('detected_label','unknown').capitalize()} | {round(doc.get('confidence',0)*100,1)}% | {doc.get('bin_label','N/A')} | {status} | {ts}"

    # Environmental impact estimate
    co2_saved = round(correct * 0.5, 1)

    return f"""
LIVE IOT DATA (Real-time from ESP32-CAM system):

OVERALL PERFORMANCE:
- Total detections: {total}
- Correct disposals: {correct} ({accuracy}%)
- Wrong disposals: {wrong} ({round(wrong/total*100,1) if total else 0}%)
- Average confidence: {avg_conf}%
- Low confidence detections (<60%): {low_conf}
- Hardest waste type to classify: {hardest}
- Estimated CO2 saved from correct disposals: {co2_saved} kg

WASTE TYPE ANALYSIS:
- Top detected types: {label_info}
- Wrong disposal rate by type: {wrong_type_info}

BIN PERFORMANCE:
- Bins with most wrong disposals: {wrong_bin_info}
- Last wrong disposal: {last_wrong_info}

TIME ANALYSIS:
- Detections today: {today_count}
- Detections yesterday: {yesterday_count}
- Peak disposal hour: {peak_hour}

LAST 5 DETECTIONS (newest first):{recent_info}
"""


def build_csv_context():
    # National averages
    avg_wrong   = round(df['wrong_disposal_rate_percent'].mean(), 1)
    avg_correct = round(df['correctly_segregated_percent'].mean(), 1)
    avg_recycle = round(df['recycling_rate_percent'].mean(), 1)
    total_co2   = round(df['co2_saved_tons'].sum(), 1)
    total_value = round(df['economic_value_LKR'].sum() / 1e6, 1)

    # Material analysis
    mat_wrong     = df.groupby('material_type')['wrong_disposal_rate_percent'].mean()
    worst_material = mat_wrong.idxmax()
    best_material  = mat_wrong.idxmin()
    worst_mat_rate = round(mat_wrong.max(), 1)
    best_mat_rate  = round(mat_wrong.min(), 1)

    # District analysis
    dist_wrong    = df.groupby('district')['wrong_disposal_rate_percent'].mean()
    worst_district = dist_wrong.idxmax()
    best_district  = dist_wrong.idxmin()

    # Urban vs rural
    urban = round(df[df['urban_rural'] == 'Urban']['correctly_segregated_percent'].mean(), 1)
    rural = round(df[df['urban_rural'] == 'Rural']['correctly_segregated_percent'].mean(), 1)

    # Trend 2015 vs 2020
    yr2015    = round(df[df['year'] == 2015]['correctly_segregated_percent'].mean(), 1)
    yr2020    = round(df[df['year'] == 2020]['correctly_segregated_percent'].mean(), 1)
    trend     = round(yr2020 - yr2015, 1)
    trend_dir = "improved" if trend > 0 else "declined"

    # Recycling rates by material
    mat_recycle      = df.groupby('material_type')['recycling_rate_percent'].mean().round(1)
    mat_recycle_info = ", ".join([f"{k}({v}%)" for k, v in mat_recycle.items()])

    # Province comparison
    prov           = df.groupby('province')['correctly_segregated_percent'].mean().round(1)
    best_province  = prov.idxmax()
    worst_province = prov.idxmin()

    return f"""
SECONDARY DATASET (Sri Lanka Waste Segregation 2015-2020, 25 districts):

NATIONAL AVERAGES:
- Average wrong disposal rate: {avg_wrong}%
- Average correctly segregated: {avg_correct}%
- Average recycling rate: {avg_recycle}%
- Total CO2 saved nationally: {total_co2} tons
- Total economic value of recycling: LKR {total_value} million

MATERIAL ANALYSIS:
- Most wrongly disposed material: {worst_material} ({worst_mat_rate}% wrong rate)
- Best segregated material: {best_material} ({best_mat_rate}% wrong rate)
- Recycling rates by material: {mat_recycle_info}

REGIONAL ANALYSIS:
- Worst performing district: {worst_district}
- Best performing district: {best_district}
- Best performing province: {best_province}
- Worst performing province: {worst_province}

BEHAVIORAL PATTERNS:
- Urban segregation accuracy: {urban}%
- Rural segregation accuracy: {rural}%
- Urban areas perform {"better" if urban > rural else "worse"} than rural by {abs(round(urban-rural,1))}%

TREND (2015-2020):
- Segregation accuracy {trend_dir} by {abs(trend)}%
- 2015 accuracy: {yr2015}%
- 2020 accuracy: {yr2020}%
"""


SYSTEM_PROMPT = """You are WasteBot, an intelligent AI assistant for the WasteVision smart waste management system.

You have access to two rich data sources:
1. LIVE IoT data from ESP32-CAM sensors (real-time detections, confidence scores, wrong disposal events)
2. Sri Lanka waste segregation dataset 2015-2020 (historical trends, regional patterns, environmental impact)

Response Rules:
- Always answer using the provided data context, never make up numbers
- Be concise and data-driven (2-5 sentences)
- For live/current questions use IoT context
- For trends/history/regions/comparisons use CSV context
- For comparison questions combine both sources
- When relevant mention environmental or economic impact
- Never say you cannot access data
- If asked to compare our system vs national average compare IoT accuracy vs CSV avg_correct
- Always be specific with numbers from the context
"""


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        body     = request.get_json()
        messages = body.get('messages', [])
        user_msg = messages[-1]['content'] if messages else ''

        # Build context from both sources
        iot_context = build_iot_context()
        csv_context = build_csv_context()
        full_context = SYSTEM_PROMPT + "\n" + iot_context + "\n" + csv_context

        # Call OpenRouter
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
            max_tokens = 250
        )

        return jsonify({'reply': response.choices[0].message.content})

    except Exception as e:
        # Fallback if LLM fails
        try:
            total    = collection.count_documents({})
            accuracy = round(collection.count_documents({'is_correct': True}) / total * 100, 1) if total else 0
            return jsonify({'reply': f"System has processed {total} detections with {accuracy}% accuracy. AI service temporarily unavailable."})
        except:
            return jsonify({'reply': "WasteBot is temporarily unavailable. Please try again."}), 500


# ─── HEALTH CHECK ─────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({
        'status'  : 'ok',
        'mongodb' : 'connected',
        'csv_rows': len(df)
    })


# ─── RUN ──────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)