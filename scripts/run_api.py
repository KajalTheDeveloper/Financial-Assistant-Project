#!/usr/bin/env python3
"""
Run API Server Script
======================

Start the FastAPI server for the RAG system.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 8080
    python scripts/run_api.py --reload
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Run the FastAPI server"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--reload", "-r",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Import uvicorn here to avoid slow startup when just checking help
    import uvicorn
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         Financial Knowledge Assistant API                 ║
╠══════════════════════════════════════════════════════════╣
║  Host:    {args.host:<45} ║
║  Port:    {args.port:<45} ║
║  Reload:  {str(args.reload):<45} ║
║  Workers: {args.workers:<45} ║
╠══════════════════════════════════════════════════════════╣
║  API Docs:    http://{args.host}:{args.port}/docs{' ' * 22}║
║  Health:      http://{args.host}:{args.port}/health{' ' * 20}║
╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app.api.routes:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
