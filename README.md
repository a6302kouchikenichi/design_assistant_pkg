# app1_design_assistant

このパッケージは app1_design_assistant を単体で動かすための最小コードです。

## 起動
```bash
python -m venv .venv
source .venv/bin/activate  # Windowsは .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
streamlit run apps/app1_design_assistant/app.py
```

## OpenAI / Azure OpenAI 切替
.env の `LLM_PROVIDER` を `openai` または `azure` にしてください。
