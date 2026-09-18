import streamlit as st
import backend

st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def get_backend():
    status_lines = []

    def progress(msg):
        status_lines.append(msg)

    backend.initialize(progress_callback=progress)
    return status_lines


with st.spinner("Loading papers and building the knowledge base (first run only)..."):
    init_log = get_backend()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def ask(query):
    st.session_state.messages.append({"role": "user", "content": query, "figures": {}})

    captured_figures = {}

    def on_chart(figures):
        captured_figures.update(figures)

    with st.spinner("Thinking..."):
        try:
            answer = backend.agent(query, on_chart=on_chart)
        except Exception as e:
            answer = f"Error: {e}"

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "figures": captured_figures,
    })


with st.sidebar:
    st.header("📚 Loaded Papers")
    for name in backend.PAPER_NAMES:
        st.markdown(f"- {name}")

    with st.expander("Indexing log"):
        for line in init_log:
            st.text(line)

    st.divider()
    st.header("Quick Actions")
    st.caption("These call the 5 available functions directly.")

    if st.button("📊 Extract experimental results", use_container_width=True):
        st.session_state.pending_query = "Extract the experimental results for all four papers."

    if st.button("📈 Compare results", use_container_width=True):
        st.session_state.pending_query = "Compare all four papers on metrics, with charts."

    if st.button("📝 Generate mini survey", use_container_width=True):
        st.session_state.pending_query = "Generate a mini survey covering all four papers."

    if st.button("🔍 Find research gaps", use_container_width=True):
        st.session_state.pending_query = (
            "Find the research gaps for all four papers: their acknowledged limitations, "
            "unsolved problems, and future work directions."
        )

    if st.button("⚖️ Compare limitations", use_container_width=True):
        st.session_state.pending_query = (
            "Compare the limitations of all four papers side by side."
        )

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.title("Research Assistant")
st.caption(f"Ask questions about: {', '.join(backend.PAPER_NAMES)}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for metric_name, fig in message.get("figures", {}).items():
            st.pyplot(fig)

if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
    ask(query)
    st.rerun()

user_input = st.chat_input("Ask your question...")
if user_input:
    ask(user_input)
    st.rerun()