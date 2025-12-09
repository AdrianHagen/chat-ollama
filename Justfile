default:  # Runs with `just` alone
    echo "Running default task"

run:
    #!/usr/bin/env bash
    if ! ollama list &>/dev/null; then
        echo "Ollama not running, starting..."
        ollama serve &
        sleep 3  # Wait for startup
    else
        echo "Ollama already running"
    fi
    uv run streamlit run src/app.py