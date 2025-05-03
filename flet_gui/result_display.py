import flet as ft
from pathlib import Path
from services import semantic_search


class ResultViewer(ft.Column):
    def __init__(self):
        super().__init__(spacing=10)
        self.results = []
        self.output_column = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
        self.export_button = ft.IconButton(
            icon=ft.icons.DOWNLOAD,
            tooltip="Export results",
            on_click=self.export_results,
            visible=False,
        )
        self.controls.extend([self.output_column, self.export_button])

    def show_results(self, results):
        self.results = results
        self.output_column.controls.clear()
        if not results:
            self.output_column.controls.append(ft.Text("No results found."))
        else:
            for r, score in results:
                self.output_column.controls.append(ft.Text(f"[{score:.3f}] {str(r)}"))
            self.export_button.visible = True
        self.update()

    def export_results(self, e):
        out_path = Path("tests/test_outputs/results_export.csv")
        semantic_search.export_results(self.results, out_path)
        self.output_column.controls.append(
            ft.Text(f"✅ Results exported to {out_path}")
        )
        self.update()
