import difflib
import re


class ZoneResolver:
    def __init__(self, df):
        pu_zones = {str(v).strip() for v in df["PU_Zone"].dropna().unique()}
        do_zones = {str(v).strip() for v in df["DO_Zone"].dropna().unique()}
        self.valid_zones = sorted(pu_zones | do_zones)
        self._zone_lookup = {self._normalize(z): z for z in self.valid_zones}

    @staticmethod
    def _normalize(zone_name: str) -> str:
        return re.sub(r"\s+", " ", zone_name.strip().lower())

    def resolve(self, zone_name: str):
        if not zone_name:
            return None

        normalized = self._normalize(zone_name)
        if normalized in self._zone_lookup:
            return self._zone_lookup[normalized]

        normalized_choices = list(self._zone_lookup.keys())
        fuzzy = difflib.get_close_matches(normalized, normalized_choices, n=1, cutoff=0.6)
        if fuzzy:
            return self._zone_lookup[fuzzy[0]]
        return None
