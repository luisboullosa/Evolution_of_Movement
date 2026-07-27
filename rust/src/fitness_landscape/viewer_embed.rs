use serde_json::Value;

#[derive(Clone, Copy)]
pub struct LiveMeta<'a> {
    pub status: &'a str,
    pub run_count: usize,
    pub total_runs: usize,
}

pub fn escape_json_for_script_embed(json: &str) -> String {
    json.replace("</", "<\\/")
}

/// Strip heavy tables from the HTML embed — heatmaps + meta only (~15 KB vs ~850 KB).
pub fn slim_for_html_embed(mut val: Value) -> Value {
    let obj = val
        .as_object_mut()
        .expect("landscape payload must be a JSON object");
    obj.insert("runs".into(), Value::Array(vec![]));
    obj.insert("cells".into(), Value::Array(vec![]));
    obj.insert("cell_diagrams".into(), Value::Array(vec![]));
    val
}

pub fn inject_live_meta(
    mut val: Value,
    live: Option<LiveMeta<'_>>,
    run_count_fallback: usize,
) -> Value {
    let obj = val
        .as_object_mut()
        .expect("landscape payload must be a JSON object");
    match live {
        Some(l) => {
            obj.insert("status".into(), Value::String(l.status.to_string()));
            obj.insert("run_count".into(), Value::from(l.run_count));
            obj.insert("total_runs".into(), Value::from(l.total_runs));
        }
        None => {
            obj.entry("status")
                .or_insert(Value::String("complete".into()));
            obj.entry("run_count")
                .or_insert(Value::from(run_count_fallback));
            obj.entry("total_runs")
                .or_insert(Value::from(run_count_fallback));
        }
    }
    val
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn embed_placeholder_replaces_script_slot_only() {
        let template = concat!(
            "<script id=\"landscape-data\" type=\"application/json\">__LANDSCAPE_JSON__</script>\n",
            "if (raw === '__LANDSCAPE_JSON__') return null;"
        );
        let embed = r#"{"heatmaps":[]}"#;
        let html = template.replacen("__LANDSCAPE_JSON__", embed, 1);
        assert!(html.contains(r#"if (raw === '__LANDSCAPE_JSON__') return null;"#));
        assert!(html.contains(r#"<script id="landscape-data" type="application/json">{"heatmaps":[]}</script>"#));
        assert_eq!(html.matches("__LANDSCAPE_JSON__").count(), 1);
    }

    #[test]
    fn script_embed_escapes_closing_tag() {
        assert_eq!(
            escape_json_for_script_embed(r#"{"x":"</script>"}"#),
            r#"{"x":"<\/script>"}"#
        );
    }

    #[test]
    fn slim_for_html_embed_strips_heavy_tables() {
        let val = json!({
            "heatmaps": [{"architecture": "MLP"}],
            "runs": [{"seed": 1}],
            "cells": [{"x_label": "nodes=1"}],
            "cell_diagrams": [{"diagram": {"nodes": []}}],
        });
        let slim = slim_for_html_embed(val);
        assert_eq!(slim["heatmaps"].as_array().unwrap().len(), 1);
        assert!(slim["runs"].as_array().unwrap().is_empty());
        assert!(slim["cells"].as_array().unwrap().is_empty());
        assert!(slim["cell_diagrams"].as_array().unwrap().is_empty());
    }

    #[test]
    fn inject_live_meta_running_sweep() {
        let val = json!({ "heatmaps": [] });
        let out = inject_live_meta(
            val,
            Some(LiveMeta {
                status: "running",
                run_count: 3,
                total_runs: 2592,
            }),
            0,
        );
        assert_eq!(out["status"], "running");
        assert_eq!(out["run_count"], 3);
        assert_eq!(out["total_runs"], 2592);
    }
}
