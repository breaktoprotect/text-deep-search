import flet as ft


class QueryBox(ft.Row):
    def __init__(self, on_search):
        super().__init__(spacing=10)
        self.expand = True
        self.query_input = ft.TextField(label="Enter your query", expand=True)
        self.search_button = ft.ElevatedButton(
            "🔍 Search", on_click=self.on_search_click
        )
        self.on_search_cb = on_search

        self.controls = [self.query_input, self.search_button]

    def on_search_click(self, e):
        if self.query_input.value.strip():
            self.on_search_cb(self.query_input.value.strip())
