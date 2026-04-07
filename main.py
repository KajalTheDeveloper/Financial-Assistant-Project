#!/usr/bin/env python3
"""
Financial Knowledge Assistant
==============================

Main entry point for the application.

Usage:
    python main.py ui          # Start Streamlit UI
    python main.py api         # Start FastAPI server
    python main.py both        # Start both (default)
    python main.py ingest      # Run document ingestion
    python main.py evaluate    # Run evaluation
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_streamlit():
    """Run Streamlit UI."""
    print("🚀 Starting Streamlit UI on http://localhost:8501")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "ui/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
    ])


def run_api():
    """Run FastAPI server."""
    print("🚀 Starting FastAPI server on http://localhost:8000")
    print("📚 API docs at http://localhost:8000/docs")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.api.routes:app",
        "--host", "localhost",
        "--port", "8000",
        "--reload",
    ])


def run_both():
    """Run both UI and API."""
    import multiprocessing
    
    print("🚀 Starting Financial Knowledge Assistant...")
    print("   UI:  http://localhost:8501")
    print("   API: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop.\n")
    
    # Run in separate processes
    api_process = multiprocessing.Process(target=run_api)
    ui_process = multiprocessing.Process(target=run_streamlit)
    
    try:
        api_process.start()
        ui_process.start()
        
        api_process.join()
        ui_process.join()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        api_process.terminate()
        ui_process.terminate()


def run_ingest():
    """Run document ingestion."""
    subprocess.run([
        sys.executable, "scripts/ingest_documents.py",
        "--input", "./data/documents",
        "--verbose",
    ])


def run_evaluate():
    """Run evaluation."""
    subprocess.run([
        sys.executable, "scripts/evaluate_rag.py",
        "--verbose",
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Financial Knowledge Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
    ui        Start Streamlit UI only
    api       Start FastAPI server only
    both      Start both UI and API (default)
    ingest    Run document ingestion
    evaluate  Run evaluation

Examples:
    python main.py               # Start both UI and API
    python main.py ui            # Start only UI
    python main.py api           # Start only API
    python main.py ingest        # Ingest documents
    python main.py evaluate      # Run evaluation
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="both",
        choices=["ui", "api", "both", "ingest", "evaluate"],
        help="Command to run (default: both)"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("""
╔══════════════════════════════════════════════════════════╗
║         🏦 Financial Knowledge Assistant                  ║
║                                                          ║
║  A RAG-powered system for financial document analysis    ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    commands = {
        "ui": run_streamlit,
        "api": run_api,
        "both": run_both,
        "ingest": run_ingest,
        "evaluate": run_evaluate,
    }
    
    commands[args.command]()


if __name__ == "__main__":
    main()
