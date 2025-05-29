from flask import Flask, request, jsonify
from youtube_backend import YouTubeAnalysisBackend
from API_KEYS import YOUTUBE_API_KEY, GOOGLE_API_KEY

app = Flask(__name__)
backend = YouTubeAnalysisBackend(YOUTUBE_API_KEY, GOOGLE_API_KEY)

@app.route('/api/initialize', methods=['POST'])
def initialize_video():
    try:
        data = request.get_json()
        video_url = data.get('video_url')
        if not video_url:
            return jsonify({'error': 'No video URL provided'}), 400

        # Initialize video analysis
        video_data = backend.initialize_video(video_url)
        
        # Convert video data to JSON-serializable format
        response_data = {
            'video_id': video_data.video_id,
            'title': video_data.title,
            'channel': video_data.channel,
            'views': video_data.views,
            'likes': video_data.likes,
            'comment_count': video_data.comment_count,
            'thumbnail_url': video_data.thumbnail_url
        }
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_comments():
    try:
        data = request.get_json()
        filter_query = data.get('filter_query')
        threshold = data.get('threshold', 0.4)
        
        if not filter_query:
            return jsonify({'error': 'No filter query provided'}), 400
            
        # Search comments
        results = backend.search_comments(
            filter_query=filter_query,
            threshold=threshold
        )
        
        # Convert results to JSON-serializable format
        response_data = {
            'category': results.category,
            'threshold': results.threshold,
            'total_comments': results.total_comments_found,
            'percentage': results.percentage_of_total,
            'clusters': [
                {
                    'id': cluster.id,
                    'size': cluster.size,
                    'summary': cluster.summary,
                    'avg_likes': cluster.avg_likes,
                    'representative_comments': cluster.representative_comments
                }
                for cluster in results.clusters
            ]
        }
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 