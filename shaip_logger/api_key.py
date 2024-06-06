from abc import ABC


class ShaipApiKey(ABC):
    _shaip_api_key = None

    @classmethod
    def set_api_key(cls, api_key):
        cls._shaip_api_key = api_key

    @classmethod
    def get_api_key(cls):
        return cls._shaip_api_key
