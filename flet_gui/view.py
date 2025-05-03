import flet as ft
from flet_gui.file_loader_ui import FileLoader
from flet_gui.query_ui import QueryBox
from flet_gui.result_display import ResultViewer
from services import semantic_search


def main(page: ft.Page):
    page.title = "🐾 Teddy Search"
    page.scroll = ft.ScrollMode.AUTO
    page.window_min_width = 900
    page.window_min_height = 700
    page.padding = 20

    app_state = {"file_id": None}

    result_viewer = ResultViewer()

    def handle_corpus_ready(file_id: str):
        app_state["file_id"] = file_id
        result_viewer.output_column.controls.clear()
        result_viewer.export_button.visible = False
        page.update()

    file_loader = FileLoader(on_ready_callback=handle_corpus_ready)

    def run_search(query: str):
        if not app_state["file_id"]:
            return
        results = semantic_search.query_corpus(
            query=query,
            file_id=app_state["file_id"],
            model_key="all-MiniLM-L6-v2",
            top_k=10,
        )
        result_viewer.show_results(results)

    query_ui = QueryBox(on_search=run_search)

    page.add(
        ft.Column(
            controls=[
                ft.Text("🐾 Teddy Search", style="headlineMedium", weight="bold"),
                ft.Row(
                    [
                        ft.ElevatedButton(
                            "📂 Select Corpus",
                            on_click=lambda e: file_loader.open(page),
                        )
                    ]
                ),
                query_ui,
                result_viewer,
            ],
            expand=True,
            spacing=20,
        )
    )
