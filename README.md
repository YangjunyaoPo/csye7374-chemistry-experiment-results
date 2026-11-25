# Chemistry Experiment Results Assistant

CSYE 7374 individual mini-project · Junyao Yang

A teaching-lab notebook records trial A at 78.0% isolated yield and trial B at 64.5%. When asked why B was lower, this command-line assistant calculates the difference, reads B's transfer observation and looks up relevant chemistry background. A third trial contains a number without a unit. The assistant can ask for the missing unit before making a comparison.

[Classroom slides](deliverables/chemistry-assistant.pptx) · [Project report](deliverables/project-report.pdf) · [Fictional lab records](examples/experiments.json)

## Check a demonstration immediately

From the repository root with Python 3.10 or later, run these commands. They use only the Python standard library and the bundled records.

```sh
python chemistry_agent.py --replay evidence/comparison.json
python chemistry_agent.py --replay evidence/missing-unit.json
python -m unittest discover -s tests -v
```

Replay shows a saved model decision, its arguments and the corresponding observation at each step. It verifies the input and note fingerprints, executes the tools again, and compares their observations and final outcome with the recording. It requires no running model. The three recordings were made with a local Qwen3-14B Q4_K_M model.

## Run the live agent

1. Clone this repository, then open a terminal in the new directory.

   ```sh
   git clone https://github.com/YangjunyaoPo/csye7374-chemistry-experiment-results.git
   cd csye7374-chemistry-experiment-results
   ```

2. In LM Studio, download the Q4_K_M build of Qwen3-14B, load it and start the server from the Developer tab. The defaults are model ID `qwen/qwen3-14b` and base URL `http://127.0.0.1:1234/v1`. Use the exact model ID displayed by your server if it differs. [Model download](https://huggingface.co/lmstudio-community/Qwen3-14B-GGUF) · [LM Studio server instructions](https://lmstudio.ai/docs/developer/core/server)

3. Run a sample or supply a question. Each live invocation asks the model for a new decision after each observation. The tool order can vary.

   ```sh
   python chemistry_agent.py --demo comparison
   python chemistry_agent.py --demo missing-unit
   python chemistry_agent.py --demo number-only
   python chemistry_agent.py --ask "What was written about the transfer in experiment B?"
   python chemistry_agent.py --demo comparison --save-trace runs/my-comparison.json
   ```

An alternative chat-completions service can be selected with `CHEM_BASE_URL`, `CHEM_MODEL`, and optionally `CHEM_API_KEY`. A remote endpoint receives the question and any record observations selected by the model. The adapter accepts HTTPS for remote services and HTTP on loopback. If the endpoint is unavailable, the live command prints an error. A saved trace has its own explicit replay command.

## How the model uses evidence

| Action | Result available for the next decision |
| --- | --- |
| `list_records` | IDs and titles to resolve which experiments are available. |
| `read_record` | Source position, recorded value, unit and notebook observations for one experiment. |
| `compare_results` | Signed target-minus-baseline difference, operand values and source positions, or a specific reason the comparison is blocked. |
| `lookup_note` | One entry from a three-topic teaching index on transfer loss, yield or percentage-point comparisons, with its source. |
| `report` / `ask` | Finish with cited evidence or request the information needed to proceed. |

The Python loop validates the model's JSON action, calls a tool and returns its result to the model. A report cites existing successful steps. For a question explicitly asking why a trial had a lower yield, the report must cite a successful comparison, the target experiment’s record and a teaching note about yield or transfer loss. The model receives feedback if its proposed report omits that evidence. The displayed quantities come directly from those tool results. A ten-decision cap bounds unproductive runs. Public purpose statements in the trace describe proposed actions. For explicit operand or baseline phrases the dispatcher checks record IDs and can return corrected arguments to the model. Other wording still depends on the model's interpretation. Live runs can take extra steps or include background even for a numerical question. The trace exposes the selected IDs and sources for inspection.

## What the recorded runs show

| User request | Actual decisions | Outcome |
| --- | --- | --- |
| Explain the change from A to B | Compare A and B → read B → retrieve the transfer note → report | B is 13.5 percentage points lower. B's notebook records visible residue. Transfer loss is one possibility. Identifying and measuring the residue would test it. |
| Compare C against A | Compare A and C → read C → ask | C records 61.0 with no unit. The run asks for C's unit and gives no difference. |
| Give only the change from A to B | Compare A and B → report | The same signed difference in two decisions. |

Anyone can check the magnitude independently in integer tenths: 645 minus 780 equals −135 tenths, or **−13.5 percentage points**. Reversing the record order gives +13.5 points. The problem case can be checked at `/records/2/result/unit` in the JSON file, where the stored value is `null`. To investigate how the result would change, make a copy, supply a verified unit and pass the copy with `--data`. The supplied examples are fictional.

The comparison reports a signed difference in the original unit, or in percentage points for percent yields. Relative percent change is outside the submission scope. The comparison requires matching comparison groups, measurement names and explicitly recorded units. It uses decimal arithmetic with bounded input precision. It does not convert units, estimate missing results, test statistical significance or determine why a reaction behaved as observed. Teaching notes use general chemistry background from [OpenStax, Chemistry 2e, section 4.4](https://openstax.org/books/chemistry-2e/pages/4-4-reaction-yields).

## Source map

- `chemistry_agent.py` provides the CLI, demo questions, trace display and readable outcomes.
- `agent_loop.py` dispatches model-selected actions, validates cited evidence and recomputes saved runs.
- `chemistry_tools.py` reads a frozen record snapshot and calculates signed differences.
- `teaching_notes.py` holds the local background index. `model_client.py` connects to the model.
- `evidence/` contains complete recorded decisions and observations. `tests/` checks tool arithmetic, model transport errors, invalid decisions, evidence references and replay consistency.

The scalar comparison follows the declared-unit and B-minus-A conventions of the separate Analysis Workbench project. This repository includes all code and data needed for its own analysis.

The course submission is based on the [development snapshot](https://github.com/YangjunyaoPo/chemistry-experiment-results-assistant/tree/238a354df71e0ca11c9d75945ed82da3f4dffe2f).
