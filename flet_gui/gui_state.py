class AppState:
    def __init__(self):
        self.file_path = None
        self.sheet = None
        self.columns = []
        self.embedding_id = None
        self.results = []

    def clear(self):
        self.__init__()
