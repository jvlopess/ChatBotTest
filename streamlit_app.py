import os
import streamlit as st
from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
from pathlib import Path
import pandas as pd

BANNER_IMAGE_PATH = "/home/jvcl/Downloads/streamlit-pandasai-main/assets/banner_cin_motorola.png"
DATA_FOLDER = "./data"

def load_file(path: str) -> pd.DataFrame:
    file_name = Path(path).name
    try:
        df = pd.read_csv(path, delimiter=';', on_bad_lines='warn', engine='python')
        return df
    except FileNotFoundError:
        st.error(f"Error: The file {path} was not found.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.warning(f"Warning: The file {file_name} is empty or contains no data rows.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading file {file_name}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_data(folder: str) -> pd.DataFrame:
    all_datasets = []
    folder_path = Path(folder)
    if not folder_path.is_dir():
        st.error(f"Error: The folder {folder} was not found.")
        return pd.DataFrame()
    csv_files = list(folder_path.glob("*.csv"))
    if not csv_files:
        st.warning(f"No CSV files found in the folder: {folder}")
        return pd.DataFrame()
    for file_path in csv_files:
        df_file = load_file(file_path)
        if not df_file.empty:
            all_datasets.append(df_file)
    if not all_datasets:
        st.warning("No data could be loaded from the CSV files, or all files were empty/problematic.")
        return pd.DataFrame()
    try:
        df = pd.concat(all_datasets, ignore_index=True)
        st.success(f"All data successfully loaded and combined. Total rows: {df.shape[0]}")
        return df
    except Exception as e:
        st.error(f"Error concatenating dataframes: {e}")
        return pd.DataFrame()

# --- Streamlit Callback and Response Parser ---
class StreamlitCallback(BaseCallback): # MODIFICADO
    def __init__(self) -> None: # Não precisa mais do 'container'
        """Initialize callback handler."""
        pass # O init pode não precisar fazer nada

    def on_code(self, response: str): # MODIFICADO
        # Não fazemos nada aqui para não mostrar o código para o usuário final
        # Se você quiser ver o código no terminal durante o desenvolvimento, pode adicionar:
        # print("----- Generated Code -----")
        # print(response)
        # print("-------------------------")
        pass

class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)
    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return
    def format_plot(self, result):
        st.image(result["value"])
        return
    def format_other(self, result):
        st.write(result["value"])
        return

# --- 1. Set Page Configuration ---
st.set_page_config(
    page_title="CIn/UFPE & Motorola - Data Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS para um banner "quadradinho pequeno" CENTRALIZADO ---
st.markdown(
    """
    <style>
    div[data-testid="stImage"] {
        margin-bottom: 20px;
    }
    div[data-testid="stImage"] > img {
        width: 100px;
        height: 100px;
        object-fit: cover;
        border-radius: 10px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 2. Display the Banner Image ---
if os.path.exists(BANNER_IMAGE_PATH):
    st.image(BANNER_IMAGE_PATH)
else:
    st.warning(f"Banner image not found at {BANNER_IMAGE_PATH}. Please check the path.")

# --- 3. Main Title and Introduction ---
st.title("📊 Intelligent Data Chatbot")
st.markdown(
    "Welcome! Interact with your dataset using natural language. "
    "This tool is a collaborative effort by CIn/UFPE and Motorola."
)
st.markdown("---")

# --- 4. Load Data ---
df = load_data(DATA_FOLDER)
if df.empty:
    st.error("No data loaded. Please ensure CSV files are present in the './data' folder and are valid.")
    st.stop()

# --- 5. Data Preview ---
with st.expander("🔎 Dataframe Preview"):
    st.write("Displaying the last 3 rows of the loaded data:")
    st.dataframe(df.tail(3))

# --- 6. Chat Interface ---
st.subheader("💬 Chat with Your Dataframe")
with st.container(border=True):
    query = st.text_area(
        "Ask your question here:",
        placeholder="e.g., What is the total sales per product? or Show me a chart of X vs Y",
        height=150,
        label_visibility="collapsed"
    )
    # pandasai_code_container = st.container() # REMOVIDO - não precisamos mais dele na UI

# --- 7. Query Processing and Response ---
if query:
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set it in your Streamlit secrets.")
    else:
        with st.spinner("Thinking... Please wait."):
            try:
                llm = OpenAI(api_token=openai_api_key)
                query_engine = SmartDataframe(
                    df,
                    config={
                        "llm": llm,
                        "response_parser": StreamlitResponse,
                        "callback": StreamlitCallback(), # MODIFICADO - não passa mais container
                        "verbose": False, # Definir como False para não imprimir logs detalhados no terminal do usuário final
                                         # Mantenha True durante o seu desenvolvimento, se útil
                        "enable_cache": True,
                    },
                )
                answer = query_engine.chat(query)
                is_plot_or_df = isinstance(answer, pd.DataFrame) or \
                                (isinstance(answer, dict) and answer.get("type") in ["plot", "dataframe"])
                if answer is not None and isinstance(answer, str) and not is_plot_or_df:
                    st.markdown(answer)
                elif answer is None and query:
                    st.info("I received your query, but I don't have a specific output to show for it right now. Try rephrasing or asking something else!")
            except Exception as e:
                st.error(f"An error occurred while processing your query: {e}")
                st.exception(e)
else:
    st.info("Please type your question in the text area above to interact with the data.")

# --- 8. Footer ---
st.markdown("---")
st.caption("Powered by CIn/UFPE & Motorola | Streamlit & PandasAI")