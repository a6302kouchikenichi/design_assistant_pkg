import sys
from pathlib import Path

# Add parent directory to path to find common module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from dotenv import load_dotenv

from common.settings import load_settings
from common.llm_client import LLMClient
from common.rag_store import build_index_from_folder, rag_answer

# Override any stale OS-level env vars so .env is the single source of truth.
load_dotenv(override=True)

SECTION_OPTIONS = [
    "1) 目的の言語化",
    "2) 指標(KPI)・評価軸の候補",
    "3) 分析メニュー案（優先度付き）",
    "4) 想定示唆と施策の方向性",
    "5) 上記を具体化するための質問集（優先度付き）",
]


def build_base_prompt(client_overview: str, expectations: str, constraints: str, notes: str, question: str) -> str:
    return f"""発注者の概要:
{client_overview}

期待:
{expectations}

制約条件:
{constraints}

補足:
{notes}

ユーザー要望:
{question}
"""


def replace_section_text(full_text: str, target_heading: str, new_section_text: str) -> str:
    if not full_text.strip():
        return new_section_text.strip()

    start = full_text.find(target_heading)
    if start == -1:
        return full_text.rstrip() + "\n\n" + new_section_text.strip()

    next_positions = []
    for heading in SECTION_OPTIONS:
        if heading == target_heading:
            continue
        pos = full_text.find(heading, start + len(target_heading))
        if pos != -1:
            next_positions.append(pos)

    end = min(next_positions) if next_positions else len(full_text)
    before = full_text[:start].rstrip()
    after = full_text[end:].lstrip()

    merged = before + "\n\n" + new_section_text.strip()
    if after:
        merged += "\n\n" + after
    return merged.strip()


st.set_page_config(page_title="分析設計アシスタント", layout="wide")
st.title("データ分析の設計アシスタント")

settings = load_settings()
llm = LLMClient(settings)

if "answer" not in st.session_state:
    st.session_state.answer = ""
if "cites" not in st.session_state:
    st.session_state.cites = []
if "last_updated_section" not in st.session_state:
    st.session_state.last_updated_section = ""

st.sidebar.header("RAG設定")
knowledge_dir = st.sidebar.text_input("ナレッジフォルダ", value="knowledge/app1")
index_dir = st.sidebar.text_input("インデックス保存先", value=".index/app1")

if st.sidebar.button("ナレッジをインデックス化"):
    with st.spinner("インデックスを作成中..."):
        try:
            n = build_index_from_folder(knowledge_dir, index_dir, llm)
            st.sidebar.success(f"インデックス作成完了: {n} chunks")
        except Exception as e:
            st.sidebar.error(f"エラー: {str(e)}")
            st.sidebar.info("APIキーやインターネット接続を確認してください")

st.subheader("入力（ヒアリング情報）")
col1, col2 = st.columns(2)
with col1:
    client_overview = st.text_area("発注者の概要", height=150, key="client_overview")
    expectations = st.text_area("期待（想定でも可）", height=150, key="expectations")
with col2:
    constraints = st.text_area("制約条件（期間・データ制約など）", height=150, key="constraints")
    notes = st.text_area("補足メモ（ヒアリング中の追記）", height=150, key="notes")

question = st.text_input(
    "いま欲しいアウトプット（例：目的・指標・分析メニュー・示唆・施策のたたき台と質問集）",
    key="question",
)

system_prompt = """あなたはコンサルタントのデータ分析設計アシスタントです。
出力は以下の見出しで日本語で、箇条書きを中心に簡潔にまとめてください。
1) 目的の言語化
2) 指標(KPI)・評価軸の候補
3) 分析メニュー案（優先度付き）
4) 想定示唆と施策の方向性
5) 上記を具体化するための質問集（優先度付き）
根拠が不十分な点は「仮」または「不明」と明記すること。
"""

if st.button("設計案を生成"):
    prompt = build_base_prompt(client_overview, expectations, constraints, notes, question)
    answer, cites = rag_answer(prompt, index_dir, llm, system_prompt, top_k=6)
    st.session_state.answer = answer
    st.session_state.cites = cites
    st.session_state.last_updated_section = "全体"

st.subheader("部分更新（項目指定）")
selected_section = st.selectbox("更新する項目", SECTION_OPTIONS, key="selected_section")
update_instruction = st.text_area(
    "更新方針（追加指示）",
    placeholder="例: 優先度を明確化し、質問集は実務で使える具体的な聞き方にしてください。",
    height=100,
    key="update_instruction",
)

if st.button("選択項目のみ更新"):
    if not st.session_state.answer:
        st.warning("先に「設計案を生成」を実行してください。")
    else:
        base_prompt = build_base_prompt(client_overview, expectations, constraints, notes, question)
        partial_system_prompt = f"""あなたはコンサルタントのデータ分析設計アシスタントです。
今回は次の1項目のみを更新してください: {selected_section}
- 出力は必ず更新対象の見出しから始めること
- 他の見出し(1-5)は出力しないこと
- 箇条書き中心で簡潔にまとめること
- 根拠が不十分な点は「仮」または「不明」と明記すること
"""
        partial_user_prompt = f"""既存の設計案（全文）:
{st.session_state.answer}

更新対象:
{selected_section}

更新方針（追加指示）:
{update_instruction or '特になし'}

ヒアリング情報:
{base_prompt}
"""
        updated_section, cites = rag_answer(partial_user_prompt, index_dir, llm, partial_system_prompt, top_k=6)
        st.session_state.answer = replace_section_text(st.session_state.answer, selected_section, updated_section)
        st.session_state.cites = cites
        st.session_state.last_updated_section = selected_section

if st.session_state.answer:
    st.subheader("出力")
    if st.session_state.last_updated_section:
        st.caption(f"最終更新: {st.session_state.last_updated_section}")
    st.write(st.session_state.answer)
    st.caption("参照（RAG）")
    st.json(st.session_state.cites)
