//! Natural language to PiQL query generation using LLMs

use piql::EvalContext;
use polars::prelude::*;

use crate::state::AppError;

// ============ Natural Language to PiQL ============

pub const PIQL_DOCS: &str = r#"PiQL is a text query language for Polars dataframes. Write queries that look like Python Polars.

## Supported Features

**DataFrame methods**
`filter`, `select`, `with_columns`, `head`, `tail`, `sort`, `drop`, `explode`, `group_by`, `join`, `rename`, `drop_nulls`, `reverse`, `top`

**Expr methods**
`alias`, `over`, `is_between`, `diff`, `shift`, `sum`, `mean`, `min`, `max`, `count`, `first`, `last`, `cast`, `fill_null`, `is_null`, `is_not_null`, `unique`, `abs`, `round`, `len`, `n_unique`, `cum_sum`, `cum_max`, `cum_min`, `rank`, `clip`, `reverse`

**pl functions**
`col`, `lit`, `when`/`then`/`otherwise`

**str namespace**
`starts_with`, `ends_with`, `to_lowercase`, `to_uppercase`, `len_chars`, `contains`, `replace`, `slice`

**dt namespace**
`year`, `month`, `day`, `hour`, `minute`, `second`

**Operators**
`+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `&`, `|`, `~`

**Sugar**
- `$col` → `pl.col("col")`
"#;

/// Example query templates - {table}, {str_col}, {num_col}, {cat_col} are placeholders
const EXAMPLE_TEMPLATES: &[(&str, &str)] = &[
    ("Filter numeric", "{table}.filter(pl.col(\"{num_col}\") > {num_val})"),
    ("Filter string equality", "{table}.filter(pl.col(\"{cat_col}\") == \"{cat_val}\")"),
    ("Select columns", "{table}.select(pl.col(\"{str_col}\"), pl.col(\"{num_col}\"))"),
    ("Sort descending", "{table}.sort(\"{num_col}\", descending=True)"),
    ("Head", "{table}.head(5)"),
    ("Group by + count", "{table}.group_by(\"{cat_col}\").agg(pl.col(\"{str_col}\").count().alias(\"count\"))"),
    ("String contains", "{table}.filter(pl.col(\"{str_col}\").str.contains(\"{str_fragment}\"))"),
];

/// Column info extracted from a dataframe
pub struct ColumnInfo {
    pub str_cols: Vec<String>,
    pub num_cols: Vec<String>,
    pub cat_col: Option<String>,  // A string column good for grouping (few unique values)
    pub sample_str: Option<String>,
    pub sample_num: Option<i64>,
    pub sample_cat: Option<String>,
}

/// Get schema info, sample data, and generate examples from actual data
pub async fn get_schema_and_examples(ctx: &EvalContext) -> (String, String) {
    // Clone dataframes for blocking task
    let dfs: Vec<(String, LazyFrame)> = ctx
        .dataframes
        .iter()
        .map(|(name, entry)| (name.clone(), entry.df.clone()))
        .collect();

    // Collect samples and column info in blocking task
    let results: Vec<(String, String, ColumnInfo)> = tokio::task::spawn_blocking(move || {
        dfs.into_iter()
            .filter_map(|(name, mut lf)| {
                let df = lf.clone().slice(0, 5).collect().ok()?;
                let sample_str = format!("{}", df);

                // Analyze columns
                let schema = lf.collect_schema().ok()?;
                let mut str_cols = Vec::new();
                let mut num_cols = Vec::new();

                for (col_name, dtype) in schema.iter() {
                    match dtype {
                        DataType::String => str_cols.push(col_name.to_string()),
                        DataType::Int64 | DataType::Int32 | DataType::Float64 => {
                            num_cols.push(col_name.to_string())
                        }
                        _ => {}
                    }
                }

                // Try to find a good categorical column and sample values
                let mut cat_col = None;
                let mut sample_cat = None;
                let mut sample_str_val = None;
                let mut sample_num = None;

                // Get sample values from first row
                if df.height() > 0 {
                    for col in &str_cols {
                        if let Ok(series) = df.column(col) {
                            if let Ok(val) = series.str() {
                                if let Some(s) = val.get(0) {
                                    if sample_str_val.is_none() {
                                        sample_str_val = Some(s.to_string());
                                    }
                                    // Check if this could be a categorical column (short values)
                                    if s.len() < 20 && cat_col.is_none() {
                                        cat_col = Some(col.clone());
                                        sample_cat = Some(s.to_string());
                                    }
                                }
                            }
                        }
                    }
                    for col in &num_cols {
                        if let Ok(series) = df.column(col) {
                            if let Ok(val) = series.i64() {
                                sample_num = val.get(0);
                                break;
                            }
                        }
                    }
                }

                Some((name, sample_str, ColumnInfo {
                    str_cols,
                    num_cols,
                    cat_col,
                    sample_str: sample_str_val,
                    sample_num,
                    sample_cat,
                }))
            })
            .collect()
    })
    .await
    .unwrap_or_default();

    // Build schema info
    let mut schema_info = String::new();
    for (name, sample, _) in &results {
        schema_info.push_str(&format!("## {}\n{}\n\n", name, sample));
    }

    // Generate examples from first suitable table
    let mut examples = String::new();
    if let Some((table, _, info)) = results.iter().find(|(_, _, i)| !i.str_cols.is_empty() && !i.num_cols.is_empty()) {
        let str_col = info.str_cols.first().map(|s| s.as_str()).unwrap_or("name");
        let num_col = info.num_cols.first().map(|s| s.as_str()).unwrap_or("id");
        let cat_col = info.cat_col.as_deref().unwrap_or(str_col);
        let num_val = info.sample_num.unwrap_or(100);
        let cat_val = info.sample_cat.as_deref().unwrap_or("value");
        let str_fragment = info.sample_str.as_deref()
            .and_then(|s| s.get(0..3))
            .unwrap_or("abc");

        for (desc, template) in EXAMPLE_TEMPLATES {
            let query = template
                .replace("{table}", table)
                .replace("{str_col}", str_col)
                .replace("{num_col}", num_col)
                .replace("{cat_col}", cat_col)
                .replace("{num_val}", &num_val.to_string())
                .replace("{cat_val}", cat_val)
                .replace("{str_fragment}", str_fragment);
            examples.push_str(&format!("# {}\n{}\n\n", desc, query));
        }
    }

    (schema_info, examples)
}

/// Build the system prompt with piql docs, examples, and schema
pub fn build_system_prompt(schema_info: &str, examples: &str) -> String {
    format!(
        r#"You are a PiQL query generator. Given a natural language question about data, respond with ONLY a valid PiQL query string.

<piql_description>
{}
</piql_description>

<examples>
{}
</examples>

<available_dataframes>
{}
</available_dataframes>

IMPORTANT:
- Respond with ONLY the PiQL query string
- Do NOT include any explanation, markdown formatting, or code blocks
- Do NOT wrap the query in quotes or backticks
- Just output the raw query that can be executed directly"#,
        PIQL_DOCS, examples, schema_info
    )
}

/// Call LLM to generate query
pub async fn generate_query(prompt: &str, system: &str) -> Result<String, AppError> {
    if let Ok(api_key) = std::env::var("OPENROUTER_API_KEY") {
        call_openrouter(&api_key, prompt, system).await
    } else {
        call_claude_cli(prompt, system).await
    }
}

async fn call_openrouter(api_key: &str, prompt: &str, system: &str) -> Result<String, AppError> {
    let client = reqwest::Client::new();
    let resp = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&serde_json::json!({
            "model": "anthropic/claude-sonnet-4",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        }))
        .send()
        .await
        .map_err(|e| AppError(format!("OpenRouter request failed: {}", e)))?;

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| AppError(format!("Failed to parse OpenRouter response: {}", e)))?;

    let query = json["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| AppError("No response content from LLM".into()))?
        .trim()
        .to_string();

    Ok(query)
}

pub async fn call_claude_cli(prompt: &str, system: &str) -> Result<String, AppError> {
    let full_prompt = format!("{}\n\nUser question: {}", system, prompt);
    let output = tokio::process::Command::new("claude")
        .args(["-p", &full_prompt])
        .output()
        .await
        .map_err(|e| AppError(format!("Failed to run claude CLI: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AppError(format!("claude CLI failed: {}", stderr)));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
