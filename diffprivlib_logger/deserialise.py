import json
from typing import Union

import diffprivlib
from sklearn.pipeline import Pipeline
from diffprivlib_logger.constants import DPL, DPL_INSTANCE, DPL_TYPE


class DiffPrivLibDecoder(json.JSONDecoder):
    """Decoder for DiffprivLib pipeline from str to model"""

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs
        )

    def object_hook(
        self, dct: dict
    ) -> Union[tuple, dict]:  # pylint: disable=E0202
        """Hook for custom deserialisation of a DiffPrivLib pipeline
        For every element, get the associated DiffPrivLib attribute.

        Args:
            dct (dict): decoded JSON object

        Raises:
            ValueError: If the serialised object is not compliant with
                        the expected format.

        Returns:
            dct (dict): value to used in place of the decoded JSON object (dct)
        """
        if "_tuple" in dct.keys():
            return tuple(dct["_items"])

        for k, v in dct.items():
            if isinstance(v, str):
                if v[:len(DPL_TYPE)] == DPL_TYPE:
                    try:
                        dct[k] = getattr(diffprivlib.models, v[len(DPL_TYPE):])
                    except Exception as e:
                        raise ValueError(e) from e

                elif v[:len(DPL_INSTANCE)] == DPL_INSTANCE:
                    try:
                        dct[k] = getattr(diffprivlib, v[len(DPL_INSTANCE):])()
                    except Exception as e:
                        raise ValueError(e) from e

        return dct


def deserialise_pipeline(diffprivlib_json: str) -> Pipeline:
    """Deserialise a DiffPriLip pipeline from string to DiffPrivLib model
    Args:
        diffprivlib_json (str): serialised DiffPrivLib pipeline

    Raises:
        ValueError: If the serialised object is not compliant with
                                    the expected format.

    Returns:
        Pipeline: DiffPrivLib pipeline
    """
    dct = json.loads(diffprivlib_json, cls=DiffPrivLibDecoder)
    if "module" in dct.keys():
        if dct["module"] != DPL:
            raise ValueError(f"JSON 'module' not equal to '{DPL}'")
    else:
        raise ValueError("Key 'module' not in submitted json request.")

    if "version" in dct.keys():
        if dct["version"] != diffprivlib.__version__:
            raise ValueError(
                f"Requested version does not match available version:"
                f" {diffprivlib.__version__}."
            )
    else:
        raise ValueError("Key 'version' not in submitted json request.")

    return Pipeline(
        [
            (val["name"], val["type"](**val["params"]))
            for val in dct["pipeline"]
        ]
    )
