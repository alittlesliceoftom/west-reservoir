# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a temperature tracking project with two main components:
1. **Streamlit Application** (`west_reservoir_tracker.py`) - A web dashboard for visualizing West Reservoir temperature data
2. **Python Environment** (`env/`) - Virtual environment for Python dependencies

## Weather API Setup (Optional)

To get real weather forecasts instead of synthetic data:

1. **Get a free API key from OpenWeatherMap:**
   - Visit https://openweathermap.org/api
   - Sign up for a free account
   - Get your API key from the dashboard

2. **Set your API key (choose one method):**
   
   **Option A: Environment Variable**
   ```bash
   export OPENWEATHER_API_KEY="your_api_key_here"
   ```
   
   **Option B: Streamlit Secrets**
   Create `.streamlit/secrets.toml`:
   ```toml
   OPENWEATHER_API_KEY = "your_api_key_here"
   ```

3. **Without API key:** The app will fall back to synthetic weather data based on historical patterns.

## Common Development Commands

### Python/Streamlit Development
```bash
# Activate virtual environment (if using env/)
source env/bin/activate  # On macOS/Linux
# or
env\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run west_reservoir_tracker.py

# Update requirements
pip freeze > requirements.txt
```

## Architecture

### Data Flow
- **Data Source**: Google Sheets CSV export (publicly accessible URL)
- **Data Processing**: Pandas for cleaning and transformation
- **Visualization**: Plotly for interactive charts
- **Web Framework**: Streamlit for the dashboard interface

### Key Components
- `west_reservoir_tracker.py`: Main Streamlit application with data loading, processing, and visualization
- `requirements.txt`: Python dependencies (Streamlit, Pandas, Plotly, Requests)
- Data is fetched in real-time from a Google Sheets CSV export URL

### Data Structure
- Temperature readings with Date and Temperature columns
- Date format: DD/MM/YYYY
- Automatic data cleaning and validation included

## Development Notes

- The application uses `@st.cache_data` decorator for performance optimization
- Error handling is implemented for data loading failures
- The app includes responsive design considerations
- Monthly aggregations and statistical calculations are performed in-memory