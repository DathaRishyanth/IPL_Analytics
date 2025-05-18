# IPL Win-Probability Engine

End-to-end project that turns raw IPL scorecards into a real-time win-probability
dashboard.

Displays stats involving all the seasons.

Predicts the best 11 given 2 teams and a particular season.

## Quickstart

```bash

python -m venv venv && source venv/Source/activate
pip install -r requirements.txt


python src/etl.py


python src/train.py


streamlit run src/app_streamlit.py
