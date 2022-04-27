from icevision import parsers


class CMParser(annotations_filepath, img_dir):
    def __init__(self, annotations_filepath, img_dir, c):
        super().__init__()
        self.parser = parsers.COCOMaskParser(annotations_filepath=annotations_filepath, img_dir=img_dir)
        return self.parser