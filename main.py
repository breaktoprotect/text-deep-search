from flet import app, AppView
from flet_gui.view import main

if __name__ == "__main__":
    # ? Native app on Windows
    # app(target=main)

    # ? For browser support
    app(target=main, view=AppView.WEB_BROWSER, port=8000)
