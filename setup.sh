mkdir -p "$HOME/.streamlit/"

# Heroku sets $PORT automatically; default locally.
PORT=${PORT:-8501}
echo "\
[server]\n\
headless = true\n\
address = \"0.0.0.0\"\n\
port = $PORT\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
" > "$HOME/.streamlit/config.toml"
