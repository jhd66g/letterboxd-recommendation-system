#!/usr/bin/env python3
"""
Flask API for the Letterboxd Recommendation System
"""

import os
import sys
from flask import Flask, request, jsonify
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine import LetterboxdRecommendationEngine

app = Flask(__name__)

# Global engine instance
engine = None

def initialize_engine():
    """Initialize the recommendation engine"""
    global engine
    if engine is None:
        engine = LetterboxdRecommendationEngine()
        if not engine.load_data():
            raise RuntimeError("Failed to load data")
    return engine

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Letterboxd Recommendation API is running'
    })

@app.route('/recommendations/<username>', methods=['GET'])
def get_recommendations(username):
    """Get recommendations for a user"""
    try:
        # Get optional parameters
        num_recommendations = request.args.get('num_recommendations', 25, type=int)
        
        # Initialize engine
        engine = initialize_engine()
        engine.num_recommendations = num_recommendations
        
        # Generate recommendations
        recommendations = engine.generate_recommendations(username)
        
        if recommendations is None:
            return jsonify({
                'error': f'Failed to generate recommendations for user {username}'
            }), 500
        
        # Convert to JSON-friendly format
        recommendations_json = []
        for _, row in recommendations.iterrows():
            recommendations_json.append({
                'rank': int(row['rank']),
                'tmdb_id': int(row['tmdb_id']),
                'title': row['title'],
                'year': str(row['year']),
                'genres': row['genres'],
                'directors': row['directors'],
                'predicted_rating': round(float(row['predicted_rating']), 2)
            })
        
        return jsonify({
            'username': username,
            'recommendations': recommendations_json,
            'total_recommendations': len(recommendations_json)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/recommendations/<username>/top/<int:top_n>', methods=['GET'])
def get_top_recommendations(username, top_n):
    """Get top N recommendations for a user"""
    try:
        # Initialize engine
        engine = initialize_engine()
        
        # Generate recommendations
        recommendations = engine.generate_recommendations(username)
        
        if recommendations is None:
            return jsonify({
                'error': f'Failed to generate recommendations for user {username}'
            }), 500
        
        # Take top N
        top_recommendations = recommendations.head(top_n)
        
        # Convert to JSON-friendly format
        recommendations_json = []
        for _, row in top_recommendations.iterrows():
            recommendations_json.append({
                'rank': int(row['rank']),
                'tmdb_id': int(row['tmdb_id']),
                'title': row['title'],
                'year': str(row['year']),
                'genres': row['genres'],
                'directors': row['directors'],
                'predicted_rating': round(float(row['predicted_rating']), 2)
            })
        
        return jsonify({
            'username': username,
            'top_n': top_n,
            'recommendations': recommendations_json
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Letterboxd Recommendation API...")
    print("Available endpoints:")
    print("  GET /health - Health check")
    print("  GET /recommendations/<username> - Get recommendations for user")
    print("  GET /recommendations/<username>/top/<n> - Get top N recommendations")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
