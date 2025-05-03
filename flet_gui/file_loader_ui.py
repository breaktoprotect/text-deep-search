import flet as ft
from pathlib import Path
import os
from services import semantic_search


class FileLoader(ft.Column):
    def __init__(self, on_ready_callback):
        super().__init__(tight=True, spacing=10)
        self.dialog = ft.AlertDialog(modal=True)
        self.on_ready = on_ready_callback

        self.sheet_dropdown = ft.Dropdown(label="Sheet", visible=False)
        self.columns_selector = ft.Column(scroll=ft.ScrollMode.AUTO, height=150)
        self.embed_button = ft.ElevatedButton(
            "🚀 Generate Embeddings", disabled=True, on_click=self.create_embeddings
        )
        self.status_text = ft.Text()

        self.file_picker = ft.FilePicker(on_result=self.on_file_selected)
        self.upload_btn = ft.ElevatedButton(
            "Upload File",
            on_click=lambda e: self.file_picker.upload(
                allow_multiple=False,
                allowed_extensions=["csv", "xlsx", "xls"],
                upload_url="upload",
            ),
        )

        self.dialog_file_path = None
        self.dialog_sheet = None
        self.dialog_columns = []

    def open(self, page: ft.Page):
        # Cached file list
        cached_files = semantic_search.list_cached_file_ids()
        file_list = ft.Column(
            [ft.Text("🗂️ Cached Files:")]
            + [
                ft.Text(f"- {meta.file_name} (id: {meta.file_id})")
                for meta in cached_files
            ],
            scroll=ft.ScrollMode.AUTO,
            height=100,
        )

        # Compose dialog content
        self.dialog.content = ft.Column(
            [
                file_list,
                self.file_picker,
                self.upload_btn,
                self.sheet_dropdown,
                ft.Text("Select Columns:", weight="bold"),
                self.columns_selector,
                self.embed_button,
                self.status_text,
            ]
        )

        # Properly attach and open dialog
        if self.dialog not in page.overlay:
            page.overlay.append(self.dialog)
        if self.file_picker not in page.overlay:
            page.overlay.append(self.file_picker)

        page.dialog = self.dialog
        self.dialog.open = True
        page.update()

    def on_file_selected(self, e: ft.FilePickerResultEvent):
        if not e.files:
            return
        file = e.files[0]
        cache_path = Path("cache") / file.name
        os.makedirs(cache_path.parent, exist_ok=True)
        with open(cache_path, "wb") as out_file:
            out_file.write(file.bytes_io.read())
        self.dialog_file_path = cache_path

        info = semantic_search.inspect_file(cache_path)
        self.sheet_dropdown.visible = False
        self.sheet_dropdown.options.clear()
        self.columns_selector.controls.clear()
        self.embed_button.disabled = True

        if info["sheets"]:
            self.sheet_dropdown.visible = True
            for sheet in info["sheets"]:
                self.sheet_dropdown.options.append(ft.dropdown.Option(sheet))
            self.sheet_dropdown.value = info["sheets"][0]
            self.sheet_dropdown.on_change = self.load_columns
            self.load_columns(None)
        elif info["columns"]:
            for col in info["columns"]:
                self.columns_selector.controls.append(
                    ft.Checkbox(label=col, value=False)
                )
            self.embed_button.disabled = False

        self.dialog.update()

    def load_columns(self, e):
        info = semantic_search.inspect_file(self.dialog_file_path)
        self.columns_selector.controls.clear()
        for col in info["columns"]:
            self.columns_selector.controls.append(ft.Checkbox(label=col, value=False))
        self.embed_button.disabled = False
        self.dialog.update()

    def create_embeddings(self, e):
        selected_cols = [
            cb.label
            for cb in self.columns_selector.controls
            if isinstance(cb, ft.Checkbox) and cb.value
        ]
        if not selected_cols:
            self.status_text.value = "⚠️ Please select at least one column."
            self.dialog.update()
            return

        self.status_text.value = "Processing embeddings... ⏳"
        self.dialog.update()

        file_id = semantic_search.prepare_corpus(
            file_path=self.dialog_file_path,
            sheet_name=(
                self.sheet_dropdown.value if self.sheet_dropdown.visible else None
            ),
            columns=selected_cols,
            model_key="all-MiniLM-L6-v2",
        )
        self.dialog.open = False
        self.on_ready(file_id)
