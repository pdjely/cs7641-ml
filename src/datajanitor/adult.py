from . import datajanitor

class Adult(datajanitor.DataJanitor):
    def __init__(self):
        url = 'http://www.google.com'
        super().__init__(name='adult', dataUrl=url)
