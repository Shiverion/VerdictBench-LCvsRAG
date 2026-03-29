import { FormEvent, useEffect, useMemo, useState } from "react";

import { AdminSummary, AnnotationStatus, Assignment, api } from "./api";

const defaultAnnotators = "annotator_1,annotator_2";
const adminEnabled = new URLSearchParams(window.location.search).get("admin") === "1";

function App() {
  const [annotatorId, setAnnotatorId] = useState("annotator_1");
  const [sessionStarted, setSessionStarted] = useState(false);
  const [assignment, setAssignment] = useState<Assignment | null>(null);
  const [status, setStatus] = useState<AnnotationStatus>("accepted");
  const [editedQuestion, setEditedQuestion] = useState("");
  const [editedAnswer, setEditedAnswer] = useState("");
  const [editedParagraphs, setEditedParagraphs] = useState("");
  const [notes, setNotes] = useState("");
  const [summary, setSummary] = useState<AdminSummary | null>(null);
  const [message, setMessage] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const [datasetPath, setDatasetPath] = useState("data/qa_dataset/qa_drafts_raw.jsonl");
  const [annotatorIds, setAnnotatorIds] = useState(defaultAnnotators);
  const [seed, setSeed] = useState(42);
  const [resetExisting, setResetExisting] = useState(false);

  const example = useMemo(
    () => ({
      accepted: {
        title: "Accepted example",
        question: "Siapa ketua panel hakim dalam perkara ini?",
        answer: "Prof. Dr. Anwar Usman, S.H., M.H.",
        why: "Use accepted when the question, answer, and supporting paragraphs are already correct and clearly grounded.",
      },
      modified: {
        title: "Modified example",
        question: "Apa dasar hukum Mahkamah berwenang mengadili perkara ini?",
        answer:
          "Pasal 24C ayat (1) UUD 1945 dan Pasal 10 ayat (1) huruf a UU Mahkamah Konstitusi.",
        why: "Use modified when the draft is basically usable but wording, scope, or supporting text needs correction.",
      },
      rejected: {
        title: "Rejected example",
        question: "Mengapa pemerintah melakukan tindakan tersebut?",
        answer: "Karena pertimbangan politik nasional.",
        why: "Use rejected when the item is unsupported, misleading, too vague, or not grounded in the verdict text.",
      },
    }),
    [],
  );

  async function loadNextTask(targetAnnotatorId = annotatorId) {
    setLoading(true);
    try {
      const result = await api.nextTask(targetAnnotatorId);
      setAssignment(result.assignment);
      if (!result.assignment) {
        setMessage("No pending assignments for this annotator.");
      } else {
        setMessage("");
        setStatus("accepted");
        setEditedQuestion(result.assignment.item.question);
        setEditedAnswer(result.assignment.item.gold_answer);
        setEditedParagraphs(result.assignment.item.gold_paragraphs.join("\n\n"));
        setNotes("");
      }
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function loadSummary() {
    setLoading(true);
    try {
      setSummary(await api.summary());
      setMessage("");
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (adminEnabled) {
      void loadSummary();
    }
  }, []);

  async function handleStart() {
    setSessionStarted(true);
    await loadNextTask();
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!assignment) return;
    setLoading(true);
    try {
      const result = await api.submitAnnotation({
        assignment_id: assignment.assignment_id,
        annotator_id: annotatorId,
        status,
        edited_question: status === "modified" ? editedQuestion : null,
        edited_gold_answer: status === "modified" ? editedAnswer : null,
        edited_gold_paragraphs:
          status === "modified"
            ? editedParagraphs.split("\n\n").map((value) => value.trim()).filter(Boolean)
            : null,
        notes: notes || null,
      });
      setAssignment(result.next_assignment);
      if (result.next_assignment) {
        setEditedQuestion(result.next_assignment.item.question);
        setEditedAnswer(result.next_assignment.item.gold_answer);
        setEditedParagraphs(result.next_assignment.item.gold_paragraphs.join("\n\n"));
        setStatus("accepted");
        setNotes("");
        setMessage("Annotation saved.");
      } else {
        setMessage("Annotation saved. No more pending assignments.");
      }
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function handleBootstrap(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    try {
      await api.bootstrap({
        dataset_path: datasetPath,
        annotator_ids: annotatorIds.split(",").map((value) => value.trim()).filter(Boolean),
        assignments_per_item: 2,
        seed,
        reset_existing: resetExisting,
      });
      await loadSummary();
      setMessage("Annotation project bootstrapped.");
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="annotator-shell">
      <header className="hero">
        <p className="eyebrow">VerdictBench Annotation</p>
        <h1>Detailed Example and Annotating</h1>
        <p className="lede">
          Read the examples first, then annotate one legal QA item at a time using your assigned annotator ID.
        </p>
      </header>

      {message ? <div className="message banner">{message}</div> : null}

      <main className="content-stack">
        <section className="panel onboarding-panel">
          <div className="section-heading">
            <h2>Annotator access</h2>
            <p>Share only the URL and the annotator ID. Annotators do not need the admin view.</p>
          </div>
          <div className="start-row">
            <input
              value={annotatorId}
              onChange={(event) => setAnnotatorId(event.target.value)}
              placeholder="Enter your annotator ID"
            />
            <button className="primary" onClick={() => void handleStart()} type="button">
              {sessionStarted ? "Reload my next task" : "Start annotating"}
            </button>
          </div>
        </section>

        <section className="panel">
          <div className="section-heading">
            <h2>Detailed Example</h2>
            <p>Use these examples as the labeling standard.</p>
          </div>
          <div className="example-grid">
            {Object.entries(example).map(([key, value]) => (
              <article className="example-card" key={key}>
                <span className={`example-badge badge-${key}`}>{value.title}</span>
                <h3>{value.question}</h3>
                <p className="example-answer">{value.answer}</p>
                <p>{value.why}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="section-heading">
            <h2>Annotating</h2>
            <p>
              Choose <strong>accepted</strong> if the draft is already correct, <strong>modified</strong> if it
              needs edits, and <strong>rejected</strong> if it is unusable or unsupported.
            </p>
          </div>

          {assignment ? (
            <form className="annotation-form" onSubmit={(event) => void handleSubmit(event)}>
              <div className="meta-grid">
                <div>
                  <span>Question ID</span>
                  <strong>{assignment.item.question_id}</strong>
                </div>
                <div>
                  <span>Verdict</span>
                  <strong>{assignment.item.verdict_id}</strong>
                </div>
                <div>
                  <span>Type</span>
                  <strong>{assignment.item.question_type}</strong>
                </div>
              </div>

              <label>
                <span>Question</span>
                <textarea
                  value={editedQuestion}
                  onChange={(event) => setEditedQuestion(event.target.value)}
                  rows={4}
                  disabled={status !== "modified"}
                />
              </label>

              <label>
                <span>Gold answer</span>
                <textarea
                  value={editedAnswer}
                  onChange={(event) => setEditedAnswer(event.target.value)}
                  rows={5}
                  disabled={status !== "modified"}
                />
              </label>

              <label>
                <span>Gold paragraphs</span>
                <textarea
                  value={editedParagraphs}
                  onChange={(event) => setEditedParagraphs(event.target.value)}
                  rows={8}
                  disabled={status !== "modified"}
                />
              </label>

              <div className="status-row">
                {(["accepted", "modified", "rejected"] as AnnotationStatus[]).map((choice) => (
                  <button
                    className={status === choice ? "status-pill active" : "status-pill"}
                    key={choice}
                    onClick={() => setStatus(choice)}
                    type="button"
                  >
                    {choice}
                  </button>
                ))}
              </div>

              <label>
                <span>Notes</span>
                <textarea
                  value={notes}
                  onChange={(event) => setNotes(event.target.value)}
                  rows={3}
                  placeholder="Optional note for edge cases"
                />
              </label>

              <button className="primary" disabled={loading} type="submit">
                {loading ? "Saving..." : "Submit annotation"}
              </button>
            </form>
          ) : (
            <div className="empty-state">Enter your annotator ID and click start to load your first task.</div>
          )}
        </section>

        {adminEnabled ? (
          <section className="panel admin-panel">
            <div className="section-heading">
              <h2>Admin</h2>
              <p>This view is only exposed when the URL includes <code>?admin=1</code>.</p>
            </div>

            <form className="bootstrap-form" onSubmit={(event) => void handleBootstrap(event)}>
              <label>
                <span>Dataset path</span>
                <input value={datasetPath} onChange={(event) => setDatasetPath(event.target.value)} />
              </label>
              <label>
                <span>Annotator IDs (comma separated)</span>
                <input value={annotatorIds} onChange={(event) => setAnnotatorIds(event.target.value)} />
              </label>
              <label>
                <span>Random seed</span>
                <input
                  type="number"
                  value={seed}
                  onChange={(event) => setSeed(Number(event.target.value))}
                />
              </label>
              <label className="checkbox">
                <input
                  checked={resetExisting}
                  onChange={(event) => setResetExisting(event.target.checked)}
                  type="checkbox"
                />
                <span>Reset existing items and assignments</span>
              </label>
              <div className="admin-actions">
                <button className="primary" disabled={loading} type="submit">
                  Bootstrap project
                </button>
                <button onClick={() => void loadSummary()} type="button">
                  Refresh summary
                </button>
              </div>
            </form>

            {summary ? (
              <div className="summary-grid">
                <article className="stat-card">
                  <span>Items</span>
                  <strong>{summary.total_items}</strong>
                </article>
                <article className="stat-card">
                  <span>Assignments</span>
                  <strong>{summary.completed_assignments}/{summary.total_assignments}</strong>
                </article>
                <article className="stat-card">
                  <span>Completed items</span>
                  <strong>{summary.completed_items}</strong>
                </article>
                <article className="stat-card">
                  <span>Cohen kappa</span>
                  <strong>{summary.cohen_kappa ?? "Not enough data"}</strong>
                </article>

                <section className="table-card">
                  <h3>Annotator load</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>Annotator</th>
                        <th>Assigned</th>
                        <th>Completed</th>
                      </tr>
                    </thead>
                    <tbody>
                      {summary.annotators.map((annotator) => (
                        <tr key={annotator.annotator_id}>
                          <td>{annotator.display_name}</td>
                          <td>{annotator.assigned}</td>
                          <td>{annotator.completed}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </section>

                <section className="table-card">
                  <h3>Agreement preview</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>Question</th>
                        <th>Verdict</th>
                        <th>Votes</th>
                        <th>Consensus</th>
                      </tr>
                    </thead>
                    <tbody>
                      {summary.agreement_items.slice(0, 8).map((vote) => (
                        <tr key={vote.question_id}>
                          <td>{vote.question_id}</td>
                          <td>{vote.verdict_id}</td>
                          <td>
                            {vote.votes.accepted}/{vote.votes.modified}/{vote.votes.rejected}
                          </td>
                          <td>{vote.is_tie ? "Disagreement" : vote.consensus_status}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </section>
              </div>
            ) : null}
          </section>
        ) : null}
      </main>
    </div>
  );
}

export default App;
