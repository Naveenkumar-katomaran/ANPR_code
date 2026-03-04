if __name__ == "__main__":
    import uvicorn
    import sys
    import os
    from pathlib import Path

    # -----------------------------
    # Resolve BASE_DIR (PyInstaller / Python)
    # -----------------------------
    if getattr(sys, "frozen", False):
        # Running as PyInstaller executable
        BASE_DIR = os.path.dirname(sys.executable)
    else:
        # Running as normal Python script
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # Fix sys.path for relative imports when running as script
        sys.path.append(BASE_DIR)

    # Optional: print for debugging
    print(f"BASE_DIR = {BASE_DIR}")

    # Run Uvicorn with FastAPI app
    uvicorn.run(
        "app.main:app",   # module:app string
        host="0.0.0.0",
        port=8000,
        reload=False
    )
