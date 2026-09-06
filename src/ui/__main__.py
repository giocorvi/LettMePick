""" Run the local testing workbench without development file watching."""

import uvicorn


def main() -> None:
    """ Serve the built frontend and inference API on localhost."""
    uvicorn.run("src.ui.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
