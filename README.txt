STREAMLIT APP - Impact of Social Media on Student Productivity

Files:
- app.py : the Streamlit app
- sample_data.csv : sample dataset used by the app
- requirements.txt : Python packages

Run locally (recommended):
1. Install Python 3.8+.
2. Create a virtual environment (optional):
   python -m venv venv
   source venv/bin/activate   (Linux/Mac) or venv\Scripts\activate (Windows)
3. Install requirements:
   pip install -r requirements.txt
4. Run the app:
   streamlit run app.py
   (The app expects sample_data.csv to be in the same folder; this is already provided.)

Deploying (quick):
- Create a free Streamlit Community Cloud account (https://streamlit.io/cloud) and push the repo to GitHub.
- Link the GitHub repo and deploy; set the main file to app.py.

Notes:
- The app allows uploading your own CSV with matching column names.
- If you want, I can also provide a packaged .zip with all files or convert this to a simple executable.
