# IPL Win-Probability Engine (2008 season)

End-to-end project that turns raw IPL scorecards into a real-time win-probability
dashboard.

## Quickstart

```bash
# 1. create & activate env  (conda, micromamba or venv)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. extract + transform
python src/etl.py

# 3. train model
python src/train.py

# 4. launch demo
streamlit run src/app_streamlit.py
