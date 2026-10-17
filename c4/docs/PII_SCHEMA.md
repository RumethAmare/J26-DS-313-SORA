# C4 PII Entity Schema and Annotation Guidelines

J26-DS-313 · Component 4 · Owner: O. S. Jayathilaka (IT23247390)

Governs the `annotations/c4/*.pii.jsonl` and `annotations/c4/*.entities.json`
layers of the shared corpus. Conforms to the project `DATA_CONTRACT.md`.

## 1. Record format

One JSON object per line in `<recording>.pii.jsonl`:

```json
{
  "ann_id":            "J26DS313_R0002_pii001",
  "recording":         "J26DS313_R0002",
  "doc_id":            "J26DS313_R0002_transcript_u001_v1",
  "field":             "text",
  "entity_id":         "J26DS313_R0002_e001",
  "label":             "PERSON",
  "start_char":        33,
  "end_char":          39,
  "surface":           "amashi",
  "script":            "LATIN",
  "role":              "ORGANISATION_REP",
  "redact":            true,
  "annotation_source": "human"
}
```

`label` carries the **entity type**. `role` carries **who the referent is**.
These are different axes and must never be interchanged — see §6.

## 2. Labels

| Label | Covers | Redact | Notes |
|---|---|---|---|
| `PERSON` | Any person's name, any script | yes | Includes agents giving a work name |
| `NIC` | National Identity Card number | yes | Old 9-digit + `V`/`X`; new 12-digit |
| `PHONE` | Telephone number | yes | `+94` or leading `0` + 9 digits |
| `ADDRESS` | A postal address of a person | yes | House number through town |
| `ACCOUNT` | Bank account / card number | yes | |
| `DOB` | Date of birth | yes | Only when it identifies a person |
| `ORG` | Organisation name | **no** | Not personal data; kept visible |
| `EMAIL` | Email address | yes | Rare in the current corpus |
| `LOCATION` | Public place, not a personal address | **no** | e.g. a branch or a town named alone |

`ADDRESS` vs `LOCATION` is the distinction that decides redaction. "No. 24,
Galle Road, Dehiwala" given as where somebody lives is `ADDRESS` and is
redacted. "Borella" named as the branch being visited is `LOCATION` and is not.
When a place is a person's residence, it is always `ADDRESS`.

`PASSPORT` is reserved by the data contract but does not occur in the corpus.

## 3. Roles

`PRIVATE_INDIVIDUAL` · `ORGANISATION_REP` · `ORGANISATION` · `PUBLIC_PLACE`

A call-centre agent giving a work name is `ORGANISATION_REP`. A customer is
`PRIVATE_INDIVIDUAL`. **Both are redacted** — role never changes whether a
person is redacted, only how the placeholder is described.

Role is a required schema field (FR7). It is not a research contribution.

## 4. Span conventions

**Offsets are Python string slice indices into the exact `text` field named by
`doc_id`**, counted in Unicode codepoints, not bytes. The rule that governs
everything:

    text[start_char:end_char] == surface

This is machine-checked by `validate.py`. A span that fails it is a defect, not
a judgement call.

### Annotate the maximal span

When a name carries an honorific or a second element, annotate the whole thing
**once**, not the parts:

    "Mr. Mifly"        ->  ONE PERSON span [10:19]
    "Mifly"            ->  do NOT additionally annotate the inner name

Nested and overlapping spans on the same mention are invalid. A token belongs to
at most one entity. Two spans were found violating this and were resolved by
keeping the longer span.

### Spoken digit sequences

The transcript renders numbers as they were said. The span covers the **entire
digit sequence including the internal spaces**:

    "inna 7 4 5 0 3 8 2 9 9 4 2 5"
             ^-- ACCOUNT [5:28], surface "7 4 5 0 3 8 2 9 9 4 2 5"

Do not annotate each digit. Do not normalise the surface — record it exactly as
it appears in the text. The normalised value belongs in the entity registry.

### Boundaries

Leading and trailing whitespace are excluded. Sentence-final punctuation is
excluded unless it is part of the identifier itself (`No.` in an address).

## 5. Entity identity and cross-script linking

`entity_id` is the mechanism behind Contribution 2 and the single most important
field in the layer.

**One real-world entity gets one `entity_id` everywhere** — across every
document (transcript, `clean_en`, `clean_si`, summary) and across both scripts.
The Latin `"Nimal"` and the Sinhala `"නිමල්"` are the same person and therefore
carry the same id:

    J26DS313_R0009_e001   "Nimal"    LATIN
    J26DS313_R0009_e001   "නිමල්"     SINHALA    <- same id, deliberately

Giving the two scripts different ids is the failure this component exists to
prevent: the person is redacted in one document and left exposed in the other.
`validate.py` check [6] enforces it — a PERSON resolved in only one script is
reported as a failure, not a warning.

Each `entity_id` is declared once in `<recording>.entities.json` with its
canonical form, label, role, pseudonym and every observed surface form.

## 6. Known defect: label/role interchange

Twelve recordings (R0008–R0013, R0015, R0019–R0021, R0025, R0026) were annotated
under an earlier convention in which a person's name was given the **role** value
as its `label`:

    WRONG   "label": "PRIVATE_INDIVIDUAL", "role": "person"
    WRONG   "label": "ORGANISATION_REP",   "role": "organization"
    RIGHT   "label": "PERSON",             "role": "PRIVATE_INDIVIDUAL"

`PRIVATE_INDIVIDUAL` and `ORGANISATION_REP` are role values and are never valid
in `label`. Both cases denote a person, so both normalise to `label: "PERSON"`
regardless of the `role` value recorded alongside — this was verified by
inspecting every affected surface form (all are personal names). The same defect
is present in the corresponding `entities.json` files, so the registry cannot be
used to repair it.

262 of 941 annotation rows are affected. Loaders **must** normalise on read
until the corpus is repaired at source.

## 7. Which documents to annotate

PII is annotated on the transcript **and** on both C2 renderings, because a name
missed in `clean_si` is just as much a disclosure as one missed in the
transcript. Each carries its own offsets against its own text:

    <rid>_transcript_u001_v1     the C1 Singlish transcript
    <rid>_c2_u001_en_v1          the C2 clean English
    <rid>_c2_u001_si_v1          the C2 clean Sinhala
    <rid>_summary_en_v1          the C2 English summary

A loader that mixes these up will silently produce garbage, since the same
`entity_id` appears in all of them at completely different offsets.

## 8. Annotation procedure

Structured identifiers (NIC, PHONE, ACCOUNT, DOB) are pre-labelled by rule and
then verified by hand. Unstructured types (PERSON, ADDRESS, ORG) are annotated
by hand throughout. Two annotators label independently and inter-annotator
agreement is computed and reported. Final judgement is never delegated to the
pre-labeller.
