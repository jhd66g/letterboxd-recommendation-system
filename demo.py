#!/usr/bin/env python3
"""
Demo script for Letterboxd movie recommendations.

Runs the API for a single user and prints a human-readable recommendation report.
"""

import argparse
import subprocess
import sys
import json
import pandas as pd
import textwrap


def run_command_silently(command):
    """Run a command and capture output without displaying it."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"


def get_recommendations(mode, epochs, username, top_n, get_metadata=False):
    """Get recommendations using the API."""
    command = f"python api.py --mode {mode} --epochs {epochs} --user {username} --top_n {top_n}"
    returncode, stdout, stderr = run_command_silently(command)
    
    if returncode != 0:
        print(f"❌ Error getting recommendations for {username}")
        if stderr:
            print(f"Error: {stderr}")
        return None
    
    # Parse recommendations from stdout
    recommendations = []
    lines = stdout.strip().split('\n')
    in_recommendations = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Start of recommendations section
        if "🎬 Top" in line and "recommendations for" in line:
            in_recommendations = True
            continue
            
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
                    rec = {
                        'rank': rank,
                        'title': title,
                        'year': year,
                        'score': score,
                        'already_reviewed': already_reviewed
                    }
                    
                    # Parse additional metadata if get_metadata is True
                    if get_metadata and i + 1 < len(lines) and i + 2 < len(lines):
                        # Look at next few lines for Director(s) and Genres
                        for j in range(1, min(4, len(lines) - i)):
                            next_line = lines[i + j].strip()
                            if next_line.startswith('Director(s):'):
                                rec['directors'] = next_line.split('Director(s): ', 1)[1].split(', ')
                            elif next_line.startswith('Genres:'):
                                rec['genres'] = next_line.split('Genres: ', 1)[1].split(', ')
                    
                    recommendations.append(rec)
            except (ValueError, IndexError):
                continue
    
    return recommendations


def load_metadata():
    """Load metadata for enriching recommendations."""
    from pathlib import Path
    
    # Find latest metadata file
    metadata_files = list(Path(".").glob("**/final_metadata_*.csv"))
    if not metadata_files:
        return None
    
    metadata_file = max(metadata_files, key=lambda p: p.stat().st_mtime)
    return pd.read_csv(metadata_file)


def print_movie(rec, meta=None, user=None, width=80):
    # Header
    year_str = f" ({rec['year']})" if rec.get('year') else ""
    already_reviewed = f"  ⭐ Already reviewed by {user}" if rec.get('already_reviewed') else ""
    print(f"\n{rec['rank']:>2}. {rec['title']}{year_str} - Score: {rec['score']:.2f}{already_reviewed}")
    print("─" * width)

    # Helper for printing aligned labels
    def print_field(label, value):
        lines = str(value).split('\n')
        print(f"    {label:<20} {lines[0]}")
        for line in lines[1:]:
            print(f"    {'':<20} {line}")

    if meta is not None:
        if pd.notna(meta.get('genres')):
            print_field("Genres:", ", ".join(eval(meta['genres'])))
        if pd.notna(meta.get('directors')):
            print_field("Director(s):", ", ".join(eval(meta['directors'])))
        if 'cast' in meta and pd.notna(meta['cast']):
            cast = eval(meta['cast'])[:5]
            cast_text = ", ".join(cast)
            wrapped_cast = textwrap.wrap(cast_text, width=width-24)  # Account for indentation
            print_field("Cast:", "\n".join(wrapped_cast))
        if pd.notna(meta.get('original_language')):
            print_field("Language:", meta['original_language'])
        if 'budget' in meta and pd.notna(meta['budget']) and meta['budget'] > 0:
            print_field("Budget:", f"${int(meta['budget']):,}")
        if 'revenue' in meta and pd.notna(meta['revenue']) and meta['revenue'] > 0:
            print_field("Revenue:", f"${int(meta['revenue']):,}")
        if 'runtime' in meta and pd.notna(meta['runtime']):
            print_field("Runtime:", f"{int(meta['runtime'])} minutes")
        if 'overview' in meta and pd.notna(meta['overview']):
            wrapped_overview = textwrap.wrap(meta['overview'], width=width-24)  # Account for indentation
            print_field("Overview:", "\n".join(wrapped_overview))
        if 'streaming_services' in meta and pd.notna(meta['streaming_services']):
            streaming = eval(meta['streaming_services']) if meta['streaming_services'] != '[]' else []
            services = ", ".join(streaming) if streaming else "Not available"
            wrapped_services = textwrap.wrap(services, width=width-24)  # Account for indentation
            print_field("Streaming:", "\n".join(wrapped_services))
    else:
        print_field("Predicted Score:", f"{rec['score']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Demo script for Letterboxd recommendations")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "cf_only", "cb_only"],
                        help="Recommendation mode (default: hybrid)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--user", type=str, required=True, help="Letterboxd username")
    parser.add_argument("--top_n", type=int, default=25, help="Number of recommendations (default: 25)")
    parser.add_argument("-unseen", action="store_true", help="Show only unseen movies")
    
    args = parser.parse_args()
    
    print(f"Getting recommendations for '{args.user}'...")
    print("=" * 60)
    
    # Determine API call size - for unseen, request much larger pool to account for reviewed movies
    if args.unseen:
        # For users with many reviews (like schaffrillas with ~1000), request significantly more
        # to ensure we get enough unseen recommendations after filtering
        api_top_n = max(500, args.top_n * 20)  # At least 500, or 20x requested amount
    else:
        api_top_n = args.top_n
    
    # Get recommendations from API
    recommendations = get_recommendations(args.mode, args.epochs, args.user, api_top_n, get_metadata=True)
    
    if not recommendations:
        print("❌ Failed to get recommendations")
        sys.exit(1)
    
    # Load metadata for detailed display
    metadata_df = load_metadata()
    
    # Filter for unseen movies if requested
    if args.unseen:
        recommendations = [rec for rec in recommendations if not rec.get('already_reviewed', False)]
        if not recommendations:
            print("❌ No unseen movies found in recommendations")
            sys.exit(0)
        # Limit to requested top_n and re-rank
        recommendations = recommendations[:args.top_n]
        for i, rec in enumerate(recommendations, 1):
            rec['rank'] = i
    
    # Display results
    total_shown = len(recommendations)
    filter_text = " (unseen only)" if args.unseen else ""
    print(f"\nTop {total_shown} recommendations for '{args.user}'{filter_text}:")
    print("=" * 60)
    
    for rec in recommendations:
        # Get metadata for this movie
        meta = None
        if metadata_df is not None:
            try:
                meta_row = metadata_df[
                    (metadata_df['title'] == rec['title']) & 
                    (metadata_df['year'] == rec['year'])
                ]
                if not meta_row.empty:
                    meta = meta_row.iloc[0]
            except Exception:
                pass
        
        # Print movie using the new formatting function
        print_movie(rec, meta, args.user)
    
    print(f"\n✅ Demo complete! Showed {total_shown} recommendations.")


if __name__ == "__main__":
    main()
