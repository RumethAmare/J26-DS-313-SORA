"""
Schema and integrity validation for the C4 PII annotation layer.

Implements the checks referenced by docs/PII_SCHEMA.md. The same validator runs
on the real corpus and on synthetic data, so a generator bug and an annotation
defect are caught by identical rules.

    [1] required fields present
    [2] label is a valid entity type (role values in `label` are the §6 defect)
    [3] role is a valid role value
    [4] offset integrity: text[start_char:end_char] == surface
    [5] no leading/trailing whitespace inside a span
    [6] every PERSON entity is observed in both LATIN and SINHALA script
    [7] no two spans in one document overlap
    [8] redact flag agrees with label and role
    [9] every entity_id is declared in the registry with the same label

Errors are defects. Warnings are legacy conventions that loaders tolerate but
that should be repaired at source.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

LABELS = {"PERSON", "NIC", "PHONE", "ADDRESS", "ACCOUNT", "DOB", "ORG",
          "EMAIL", "LOCATION", "PASSPORT"}
ROLES = {"PRIVATE_INDIVIDUAL", "ORGANISATION_REP", "ORGANISATION", "PUBLIC_PLACE"}
SCRIPTS = {"LATIN", "SINHALA"}

# Entity types that are not personal data and stay visible (§2).
NOT_REDACTED = {"ORG", "LOCATION"}

# §6: role values written into `label` under the earlier convention. Both
# denote a person, so both normalise to PERSON.
_ROLE_AS_LABEL = {"PRIVATE_INDIVIDUAL", "ORGANISATION_REP"}

# Lower-case role values from the earlier convention. They are reported, not
# repaired: "identifier" does not say whose identifier it is, and guessing an
# owner would put an unverified claim into the data.
LEGACY_ROLES = {"person", "organization", "identifier"}

REQUIRED = ("ann_id", "doc_id", "entity_id", "label",
            "start_char", "end_char", "surface", "script", "role", "redact")
# Required by the schema but derivable from doc_id, so its absence is reported
# without blocking the checks that matter (525 legacy rows lack it).
DERIVABLE = ("recording",)


def normalise_label(row: dict) -> dict:
    """Return a copy of `row` with the §6 label/role interchange repaired."""
    if row.get("label") in _ROLE_AS_LABEL:
        row = dict(row, label="PERSON")
    return row


def expected_redact(label: str, role: str | None) -> bool:
    """
    ORG and LOCATION are never redacted. An identifier owned by an organisation
    (a published hotline) is not personal data either. Everything else is.
    """
    return label not in NOT_REDACTED and role != "ORGANISATION"


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_recording(rows: list[dict], texts: dict[str, str],
                       registry: dict | None = None) -> Report:
    """
    Validate one recording.

    rows      the <recording>.pii.jsonl records
    texts     doc_id -> the exact text the offsets index into
    registry  the parsed <recording>.entities.json, or None to skip check [9]
    """
    rep = Report()
    scripts_seen: dict[str, set[str]] = defaultdict(set)
    person_ids: set[str] = set()
    spans_by_doc: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    declared = ({e["entity_id"]: e for e in registry.get("entities", [])}
                if registry is not None else None)

    for raw in rows:
        aid = raw.get("ann_id", "<no ann_id>")

        missing = [k for k in REQUIRED if k not in raw]
        if missing:
            rep.errors.append(f"[1] {aid}: missing fields {missing}")
            continue
        for k in DERIVABLE:
            if k not in raw:
                rep.warnings.append(f"[1] {aid}: missing derivable field {k!r}")

        if raw["label"] in _ROLE_AS_LABEL:
            rep.warnings.append(f"[2] {aid}: role value {raw['label']!r} in label "
                                f"(section 6 defect, normalised to PERSON)")
        row = normalise_label(raw)
        label, role = row["label"], row["role"]

        if label not in LABELS:
            rep.errors.append(f"[2] {aid}: invalid label {label!r}")
        if role in LEGACY_ROLES:
            rep.warnings.append(f"[3] {aid}: legacy role {role!r}")
        elif role not in ROLES:
            rep.errors.append(f"[3] {aid}: invalid role {role!r}")
        if row["script"] not in SCRIPTS:
            rep.errors.append(f"[3] {aid}: invalid script {row['script']!r}")

        start, end, surface = row["start_char"], row["end_char"], row["surface"]
        text = texts.get(row["doc_id"])
        if text is None:
            rep.errors.append(f"[4] {aid}: unknown doc_id {row['doc_id']!r}")
        elif not (0 <= start < end <= len(text)) or text[start:end] != surface:
            got = text[start:end] if 0 <= start < end <= len(text) else "<out of range>"
            rep.errors.append(f"[4] {aid}: text[{start}:{end}] is {got!r}, "
                              f"surface is {surface!r}")

        if surface != surface.strip():
            rep.errors.append(f"[5] {aid}: surface {surface!r} has edge whitespace")

        if role not in LEGACY_ROLES and row["redact"] != expected_redact(label, role):
            rep.errors.append(f"[8] {aid}: redact={row['redact']} for {label}/{role}")

        spans_by_doc[row["doc_id"]].append((start, end, aid))
        scripts_seen[row["entity_id"]].add(row["script"])
        if label == "PERSON":
            person_ids.add(row["entity_id"])

        if declared is not None:
            ent = declared.get(row["entity_id"])
            if ent is None:
                rep.errors.append(f"[9] {aid}: entity_id {row['entity_id']!r} "
                                  f"not in registry")
            elif normalise_label(ent)["label"] != label:
                rep.errors.append(f"[9] {aid}: label {label} disagrees with "
                                  f"registry label {ent['label']}")

    for eid in sorted(person_ids):
        missing_scripts = SCRIPTS - scripts_seen[eid]
        if missing_scripts:
            rep.errors.append(f"[6] {eid}: PERSON never observed in "
                              f"{sorted(missing_scripts)} script")

    for doc_id, spans in spans_by_doc.items():
        spans.sort()
        for (s1, e1, a1), (s2, e2, a2) in zip(spans, spans[1:]):
            if s2 < e1:
                rep.errors.append(f"[7] {doc_id}: {a1} [{s1}:{e1}] overlaps "
                                  f"{a2} [{s2}:{e2}]")
    return rep
