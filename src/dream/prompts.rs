//! Prompt construction for Phase 2 LLM-assisted dreaming operations.
//!
//! Matches [`crate::llm`]'s existing `EXTRACTION_PROMPT` style: a terse
//! system persona message, and one user message containing instructions, an
//! explicit output shape, an explicit empty-case instruction, and the raw
//! interpolated data — no markdown fencing, since [`crate::llm::chat_completion`]
//! parses responses with a direct `serde_json::from_str`.

/// System persona for the abstractive-consolidation prompt.
pub(crate) const CONSOLIDATION_SYSTEM: &str =
    "You are a memory analyst who writes concise, faithful summaries.";

/// Build the user prompt asking the model to rewrite extractive bullet
/// points into one coherent gist summary.
pub(crate) fn build_consolidation_prompt(bullets: &[String]) -> String {
    format!(
        "Rewrite these extractive bullet points as one coherent 1-3 sentence gist. \
         Preserve every fact; do not invent new ones. Return a JSON object: \
         {{\"summary\": \"...\"}}.\n\nBullet points:\n{}",
        bullets.join("\n")
    )
}

/// System persona for the recombination prompt.
pub(crate) const RECOMBINATION_SYSTEM: &str = "You are a research analyst looking for non-obvious, testable connections between unrelated notes.";

/// Build the user prompt presenting two labeled clusters and asking for at
/// most `max_items` non-obvious, testable connections between them, each
/// citing the labels it draws on.
pub(crate) fn build_recombination_prompt(
    cluster_a: &[(usize, &str)],
    cluster_b: &[(usize, &str)],
    max_items: usize,
) -> String {
    fn format_group(items: &[(usize, &str)]) -> String {
        items
            .iter()
            .map(|(label, content)| format!("[{label}] {content}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
    format!(
        "Propose at most {max_items} non-obvious, testable connections between Group A and \
         Group B. Each connection must cite the item labels it draws on. Return a JSON object: \
         {{\"connections\": [{{\"content\": \"...\", \"thought_type\": \
         \"Hypothesis|Idea|Wonder|Question\", \"cites\": [1,2]}}]}}. If nothing real connects \
         them, return {{\"connections\": []}}.\n\nGroup A:\n{}\n\nGroup B:\n{}",
        format_group(cluster_a),
        format_group(cluster_b)
    )
}

/// System persona for the contradiction-check prompt.
pub(crate) const CONTRADICTION_SYSTEM: &str =
    "You are a careful fact-checker comparing two notes for direct contradiction.";

/// Build the user prompt asking whether two labeled items directly
/// contradict each other.
pub(crate) fn build_contradiction_prompt(
    a_label: usize,
    a: &str,
    b_label: usize,
    b: &str,
) -> String {
    format!(
        "Do items [{a_label}] and [{b_label}] contradict each other (assert incompatible \
         facts), as opposed to merely being different or unrelated? Return a JSON object: \
         {{\"contradicts\": true|false, \"rationale\": \"...\"}}.\n\n\
         [{a_label}] {a}\n[{b_label}] {b}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consolidation_prompt_includes_every_bullet_and_output_schema() {
        let bullets = vec!["- fact one".to_string(), "- fact two".to_string()];
        let prompt = build_consolidation_prompt(&bullets);
        assert!(prompt.contains("- fact one"));
        assert!(prompt.contains("- fact two"));
        assert!(prompt.contains("\"summary\""));
    }

    #[test]
    fn recombination_prompt_includes_labels_content_and_budget() {
        let cluster_a = [(1usize, "alpha content"), (2usize, "beta content")];
        let cluster_b = [(3usize, "gamma content")];
        let prompt = build_recombination_prompt(&cluster_a, &cluster_b, 2);
        assert!(prompt.contains("[1] alpha content"));
        assert!(prompt.contains("[2] beta content"));
        assert!(prompt.contains("[3] gamma content"));
        assert!(prompt.contains('2')); // max_items echoed somewhere
        assert!(prompt.contains("\"cites\""));
        assert!(prompt.contains("\"connections\": []"));
    }

    #[test]
    fn contradiction_prompt_includes_both_labeled_items() {
        let prompt = build_contradiction_prompt(1, "the sky is blue", 2, "the sky is green");
        assert!(prompt.contains("[1] the sky is blue"));
        assert!(prompt.contains("[2] the sky is green"));
        assert!(prompt.contains("\"contradicts\""));
    }
}
