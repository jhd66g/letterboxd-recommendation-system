#!/bin/bash

# Letterboxd Data Pipeline Runner
# This script orchestrates the complete data collection and processing pipeline

set -e  # Exit on any error

# Default values
NUM_USERS=${1:-20}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="data_${NUM_USERS}_${TIMESTAMP}"
PYTHON_PATH="../.venv/bin/python"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "get_users.py" ]; then
    error "Please run this script from the data_pipeline directory"
    exit 1
fi

# Check if Python virtual environment exists
if [ ! -f "$PYTHON_PATH" ]; then
    error "Python virtual environment not found at $PYTHON_PATH"
    exit 1
fi

# Create output directory
log "Creating output directory: ../$OUTPUT_DIR"
mkdir -p "../$OUTPUT_DIR"

# Archive old data if it exists
if [ -d "../data" ]; then
    ARCHIVE_DIR="../archive/data_archived_$(date +"%Y%m%d_%H%M%S")"
    log "Archiving existing data to $ARCHIVE_DIR"
    mkdir -p "$ARCHIVE_DIR"
    mv ../data/* "$ARCHIVE_DIR/" 2>/dev/null || true
fi

# Archive old data_* folders
for old_data_dir in ../data_*; do
    if [ -d "$old_data_dir" ] && [ "$(basename "$old_data_dir")" != "$OUTPUT_DIR" ]; then
        BASENAME=$(basename "$old_data_dir")
        ARCHIVE_TARGET="../archive/${BASENAME}_archived_$(date +"%Y%m%d_%H%M%S")"
        log "Archiving old data folder: $BASENAME to archive/"
        mkdir -p "$ARCHIVE_TARGET"
        mv "$old_data_dir"/* "$ARCHIVE_TARGET/" 2>/dev/null || true
        rmdir "$old_data_dir" 2>/dev/null || true
    fi
done

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Letterboxd Data Pipeline Started${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Target users: ${YELLOW}$NUM_USERS${NC}"
echo -e "Output directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Timestamp: ${YELLOW}$TIMESTAMP${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Step 1: Get Users
log "Step 1/5: Getting popular Letterboxd users..."
USERS_FILE="../$OUTPUT_DIR/letterboxd_users_$TIMESTAMP.csv"

$PYTHON_PATH get_users.py \
    --max-users $NUM_USERS \
    --time-period "this/week" \
    --output "$USERS_FILE" \
    --usernames-only

if [ ! -f "$USERS_FILE" ]; then
    error "Failed to create users file"
    exit 1
fi

USER_COUNT=$(tail -n +2 "$USERS_FILE" | wc -l | tr -d ' ')
success "Collected $USER_COUNT users"

# Step 2: Scrape Reviews
log "Step 2/5: Scraping reviews from Letterboxd..."
REVIEWS_FILE="../$OUTPUT_DIR/letterboxd_reviews_$TIMESTAMP.csv"

$PYTHON_PATH scraper.py \
    --input "$USERS_FILE" \
    --output "$REVIEWS_FILE" \
    --explicit-only

if [ ! -f "$REVIEWS_FILE" ]; then
    error "Failed to create reviews file"
    exit 1
fi

REVIEW_COUNT=$(tail -n +2 "$REVIEWS_FILE" | wc -l | tr -d ' ')
success "Collected $REVIEW_COUNT reviews"

# Step 3: Extract Unique Movies
log "Step 3/5: Extracting unique movies..."
UNIQUE_MOVIES_FILE="../$OUTPUT_DIR/unique_movies_$TIMESTAMP.csv"

$PYTHON_PATH unique_movies_extractor.py \
    "$REVIEWS_FILE" \
    "$UNIQUE_MOVIES_FILE"

if [ ! -f "$UNIQUE_MOVIES_FILE" ]; then
    error "Failed to create unique movies file"
    exit 1
fi

MOVIE_COUNT=$(tail -n +2 "$UNIQUE_MOVIES_FILE" | wc -l | tr -d ' ')
success "Extracted $MOVIE_COUNT unique movies"

# Step 4: Fetch TMDB Metadata
log "Step 4/5: Fetching TMDB metadata..."
METADATA_FILE="../$OUTPUT_DIR/movie_metadata_$TIMESTAMP.csv"

$PYTHON_PATH tmdb_fetcher.py \
    "$UNIQUE_MOVIES_FILE" \
    "$METADATA_FILE"

if [ ! -f "$METADATA_FILE" ]; then
    error "Failed to create metadata file"
    exit 1
fi

METADATA_COUNT=$(tail -n +2 "$METADATA_FILE" | wc -l | tr -d ' ')
success "Fetched metadata for $METADATA_COUNT movies"

# Step 5: Clean and Process Data
log "Step 5/5: Cleaning and processing final dataset..."
FINAL_REVIEWS_FILE="../$OUTPUT_DIR/final_reviews_$TIMESTAMP.csv"
FINAL_METADATA_FILE="../$OUTPUT_DIR/final_metadata_$TIMESTAMP.csv"

$PYTHON_PATH data_cleaner.py \
    "$REVIEWS_FILE" \
    "$METADATA_FILE" \
    "$FINAL_REVIEWS_FILE" \
    "$FINAL_METADATA_FILE"

if [ ! -f "$FINAL_REVIEWS_FILE" ] || [ ! -f "$FINAL_METADATA_FILE" ]; then
    error "Failed to create final cleaned files"
    exit 1
fi

FINAL_REVIEW_COUNT=$(tail -n +2 "$FINAL_REVIEWS_FILE" | wc -l | tr -d ' ')
FINAL_METADATA_COUNT=$(tail -n +2 "$FINAL_METADATA_FILE" | wc -l | tr -d ' ')
success "Final dataset: $FINAL_REVIEW_COUNT reviews, $FINAL_METADATA_COUNT movies"

# Create pipeline summary
SUMMARY_FILE="../$OUTPUT_DIR/pipeline_summary_$TIMESTAMP.json"
cat > "$SUMMARY_FILE" << EOF
{
    "pipeline_run": {
        "timestamp": "$TIMESTAMP",
        "target_users": $NUM_USERS,
        "execution_time": "$(date)",
        "output_directory": "$OUTPUT_DIR"
    },
    "data_collected": {
        "users": $USER_COUNT,
        "raw_reviews": $REVIEW_COUNT,
        "unique_movies": $MOVIE_COUNT,
        "movies_with_metadata": $METADATA_COUNT,
        "final_reviews": $FINAL_REVIEW_COUNT,
        "final_movies": $FINAL_METADATA_COUNT
    },
    "files_created": {
        "users": "letterboxd_users_$TIMESTAMP.csv",
        "reviews": "letterboxd_reviews_$TIMESTAMP.csv",
        "unique_movies": "unique_movies_$TIMESTAMP.csv",
        "metadata": "movie_metadata_$TIMESTAMP.csv",
        "final_reviews": "final_reviews_$TIMESTAMP.csv",
        "final_metadata": "final_metadata_$TIMESTAMP.csv"
    }
}
EOF

# Final summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}   Pipeline Completed Successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "📊 ${BLUE}Data Collection Summary:${NC}"
echo -e "   👤 Users collected: ${YELLOW}$USER_COUNT${NC}"
echo -e "   📝 Reviews scraped: ${YELLOW}$REVIEW_COUNT${NC}"
echo -e "   🎬 Unique movies: ${YELLOW}$MOVIE_COUNT${NC}"
echo -e "   📊 Movies with metadata: ${YELLOW}$METADATA_COUNT${NC}"
echo -e "   ✨ Final reviews: ${YELLOW}$FINAL_REVIEW_COUNT${NC}"
echo -e "   🎭 Final movies: ${YELLOW}$FINAL_METADATA_COUNT${NC}"
echo -e "\n📁 ${BLUE}Output Location:${NC} ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "📋 ${BLUE}Pipeline Summary:${NC} ${YELLOW}$SUMMARY_FILE${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Set the new data directory as the default
if [ -d "../data" ]; then
    rm -rf "../data"
fi
ln -sf "$OUTPUT_DIR" "../data"
success "Created symlink: data -> $OUTPUT_DIR"

log "Pipeline completed successfully! 🚀"
