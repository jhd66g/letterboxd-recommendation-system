#!/usr/bin/env python3
"""
Test script for Letterboxd Recommendation API

Batch-runs the API for several usernames and reports results; also runs an evaluation pass.
"""

import argparse
import subprocess
import sys
from typing import List, Dict, Any
import json
import os


def run_command_silently(command: List[str]) -> tuple[int, str, str]:
    """Run a command and capture output silently."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def parse_recommendations_output(stdout: str) -> tuple[bool, List[Dict[str, Any]]]:
    """Parse API output to extract user status and recommendations."""
    lines = stdout.strip().split('\n')
    
    # Check if user was newly added or existing
    newly_added = "Scraping new user:" in stdout or "Added user" in stdout
    
    # Extract recommendations from the formatted output
    recommendations = []
    in_recommendations = False
    
    for line in lines:
        line = line.strip()
        
        # Start of recommendations section
        if "🎬 Top" in line and "recommendations for" in line:
            in_recommendations = True
            continue
            
        # End of recommendations section
        if in_recommendations and line.startswith('%'):
            break
            
        # Parse recommendation lines
        if in_recommendations and line and not line.startswith('=') and not line.startswith('    '):
            try:
                # Format: " 1. Movie Title (2022) - Score: 8.35" or " 1. Movie Title (2022) - Score: 8.35 ⭐"
                if '. ' in line and ' - Score: ' in line:
                    # Check for already reviewed indicator
                    already_reviewed = ' ⭐' in line
                    if already_reviewed:
                        line = line.replace(' ⭐', '')  # Remove indicator for parsing
                    
                    parts = line.split('. ', 1)[1]  # Remove rank number
                    title_year, score_part = parts.rsplit(' - Score: ', 1)
                    score = float(score_part)
                    
                    # Extract title and year
                    if '(' in title_year and title_year.endswith(')'):
                        title = title_year.rsplit(' (', 1)[0]
                        year = int(title_year.rsplit(' (', 1)[1][:-1])
                    else:
                        title = title_year
                        year = None
                    
                    rank = len(recommendations) + 1
                    recommendations.append({
                        'rank': rank,
                        'title': title,
                        'year': year,
                        'score': score,
                        'already_reviewed': already_reviewed
                    })
                    
                    # Only get top 5
                    if len(recommendations) >= 5:
                        break
            except (ValueError, IndexError):
                continue
    
    return newly_added, recommendations


def test_user(username: str, mode: str = "hybrid") -> tuple[bool, List[Dict[str, Any]]]:
    """Test API for a single user and return status and top-5 recommendations."""
    print(f"Testing {username}...", end=" ", flush=True)
    
    command = [
        sys.executable, "api.py",
        "--mode", mode,
        "--epochs", "10",
        "--user", username,
        "--top_n", "5"
    ]
    
    returncode, stdout, stderr = run_command_silently(command)
    
    if returncode != 0:
        print("❌ FAILED")
        print(f"  Error: {stderr.strip() if stderr else 'Unknown error'}")
        return False, []
    
    newly_added, recommendations = parse_recommendations_output(stdout)
    status = "NEW" if newly_added else "EXISTING"
    print(f"✅ {status}")
    
    return newly_added, recommendations


def run_evaluation(mode: str = "hybrid") -> Dict[str, float]:
    """Run model evaluation and return metrics."""
    print(f"\nEvaluating for 30 epochs...")
    
    command = [
        sys.executable, "recommendation_engine.py",
        "--mode", mode,
        "--train",
        "--epochs", "30",
        "--evaluate"
    ]
    
    returncode, stdout, stderr = run_command_silently(command)
    
    if returncode != 0:
        print("❌ Evaluation failed")
        return {}
    
    # Try to extract JSON metrics from output (handle multi-line JSON)
    stdout_text = stdout.strip()
    if '{' in stdout_text and '"mode"' in stdout_text and '"precision_at_k"' in stdout_text:
        try:
            # Find the JSON block
            start_idx = stdout_text.find('{')
            end_idx = stdout_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_text = stdout_text[start_idx:end_idx]
                metrics = json.loads(json_text)
                return metrics
        except json.JSONDecodeError:
            pass
    
    return {}


def main():
    parser = argparse.ArgumentParser(description="Batch test the recommendation API")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "cf_only", "cb_only"],
                        help="Recommendation mode (default: hybrid)")
    
    args = parser.parse_args()
    
    # Test usernames from README
    usernames = [
        "schaffrillas",
        "davidehrlich", 
        "kurstboy",
        "jaaackd",
        "aidandking08"
    ]
    
    print(f"🧪 Testing API in {args.mode} mode")
    print("=" * 60)
    
    # Test each user
    all_results = {}
    for username in usernames:
        newly_added, recommendations = test_user(username, args.mode)
        all_results[username] = {
            'newly_added': newly_added,
            'recommendations': recommendations
        }
    
    # Display results
    print("\n📊 Results Summary")
    print("=" * 60)
    
    for username, result in all_results.items():
        status = "NEW USER" if result['newly_added'] else "EXISTING USER"
        print(f"\n👤 {username} ({status})")
        print("─" * 40)
        
        if result['recommendations']:
            for rec in result['recommendations']:
                year_str = f" ({rec['year']})" if rec['year'] else ""
                reviewed_str = " ⭐" if rec.get('already_reviewed', False) else ""
                print(f"  {rec['rank']}. {rec['title']}{year_str} - Score: {rec['score']:.2f}{reviewed_str}")
        else:
            print("  ❌ No recommendations generated")
    
    # Run evaluation
    print("\n" + "=" * 60)
    metrics = run_evaluation(args.mode)
    
    if metrics:
        print("📈 Evaluation Metrics")
        print("─" * 40)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    else:
        print("❌ Evaluation metrics not available")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()
