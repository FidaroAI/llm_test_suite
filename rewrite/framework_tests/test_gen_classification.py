import json

from llmeval.generation.classification import (
    UNCLASSIFIED,
    labels_for,
    load_classifications,
    prompt_key,
)


def _write(dir_, suite, mapping):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{suite}.json").write_text(json.dumps({"classifications": mapping}))


def test_prompt_key_is_trimmed_sha1():
    assert prompt_key(" hi ") == prompt_key("hi")
    assert prompt_key("a") != prompt_key("b")


def test_labels_looked_up_by_prompt_hash(tmp_path):
    _write(tmp_path, "agentharm_refusal",
           {prompt_key("Do X"): {"request_type": "coding", "domain": "technology_ai"}})
    mapping = load_classifications("agentharm_refusal", str(tmp_path))
    labels = labels_for("Do X", mapping)
    assert labels == {"request_type": "coding", "domain": "technology_ai"}


def test_unknown_prompt_falls_back_to_unclassified(tmp_path):
    _write(tmp_path, "s", {})
    mapping = load_classifications("s", str(tmp_path))
    assert labels_for("never seen", mapping) == {
        "request_type": UNCLASSIFIED, "domain": UNCLASSIFIED}


def test_missing_file_is_empty_mapping(tmp_path):
    mapping = load_classifications("absent", str(tmp_path))
    assert mapping == {}
    assert labels_for("q", mapping)["domain"] == UNCLASSIFIED
