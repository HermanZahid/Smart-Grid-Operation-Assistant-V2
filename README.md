# Renewable Grid Operations Agent

A simple Streamlit portfolio application demonstrating Generative AI and agentic AI for renewable-integrated power-system operation.

## Files
- `app.py` — complete application
- `requirements.txt` — dependencies
- `grid_input_24h.csv` — fixed 24-hour input dataset
- `README.md` — deployment notes

## Features
- Grid Health dashboard
- Automatic Critical Hours detection
- Intelligent battery analysis
- Power-balance and battery charts
- Battery what-if simulator
- Base-case vs scenario comparison
- Groq tool-calling agent with visible tool activity
- AI Operator Chat with suggested questions

## Deploy on Streamlit
1. Put these files in one GitHub repository.
2. Create a Streamlit app pointing to `app.py`.
3. In Streamlit Secrets add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

4. Deploy/redeploy.
5. Upload `grid_input_24h.csv` in the application.

The Groq model used is `openai/gpt-oss-120b`.

## Important
The numerical power-system calculations are deterministic Python calculations. The Groq model selects tools and explains the calculated results. This is an educational aggregated energy-balance model, not a power-flow, OPF, SCADA, or safety-critical controller.
