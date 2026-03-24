use crate::schema::{build_schema, StickerSchema};
use crate::tokenizers::{caption_analyzer, semantic_analyzer};
use anyhow::Result;
use axum::{extract::State, routing::{get, post}, Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tantivy::collector::TopDocs;
use tantivy::query::{AllQuery, BooleanQuery, Occur, Query, QueryParser, TermQuery};
use tantivy::schema::{IndexRecordOption, TantivyDocument, Value};
use tantivy::{Index, ReloadPolicy, Term};

#[derive(Clone)]
pub struct AppState {
    pub index: Index,
    pub schema: StickerSchema,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub schema_version: String,
    pub tokenizers: Vec<String>,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub caption_query_text: String,
    pub sticker_query_text: String,
    pub caption_importance: Option<String>,
    pub allow_animation: Option<bool>,
    pub top_k: Option<usize>,
}

#[derive(Serialize, Clone)]
pub struct SearchHit {
    pub sticker_id: String,
    pub score: f32,
    pub fields: Vec<String>,
    pub debug: serde_json::Value,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub hits: Vec<SearchHit>,
}

#[derive(Default)]
struct MutableHit {
    sticker_id: String,
    score: f32,
    fields: Vec<String>,
    caption_lexical: f32,
    sticker_lexical: f32,
}

pub async fn serve(index_dir: PathBuf, port: u16) -> Result<()> {
    let schema = build_schema();
    let index = Index::open_in_dir(index_dir)?;
    index.tokenizers().register("caption", caption_analyzer());
    index.tokenizers().register("semantic", semantic_analyzer());
    let state = Arc::new(AppState { index, schema });
    let app = Router::new()
        .route("/health", get(health))
        .route("/search", post(search))
        .with_state(state);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}


async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        service: "tantivy_retriever".to_string(),
        schema_version: "human_sticker_v1".to_string(),
        tokenizers: vec!["caption".to_string(), "semantic".to_string()],
    })
}
async fn search(State(state): State<Arc<AppState>>, Json(req): Json<SearchRequest>) -> Json<SearchResponse> {
    let reader = state.index.reader_builder().reload_policy(ReloadPolicy::OnCommitWithDelay).try_into().unwrap();
    let searcher = reader.searcher();
    let top_k = req.top_k.unwrap_or(50).clamp(1, 200);
    let caption_importance = req.caption_importance.unwrap_or_else(|| "prefer".to_string());
    let allow_animation = req.allow_animation.unwrap_or(false);

    let caption_query = build_caption_query(&state, &req.caption_query_text, &caption_importance, allow_animation);
    let sticker_query = build_sticker_query(&state, &req.sticker_query_text, allow_animation);

    let mut merged: HashMap<String, MutableHit> = HashMap::new();
    collect_hits(&searcher, &state.schema, caption_query.as_ref(), top_k, "caption_lexical", &mut merged);
    collect_hits(&searcher, &state.schema, sticker_query.as_ref(), top_k, "sticker_lexical", &mut merged);

    let mut hits: Vec<SearchHit> = merged
        .into_values()
        .map(|hit| SearchHit {
            sticker_id: hit.sticker_id,
            score: hit.score,
            fields: hit.fields,
            debug: serde_json::json!({
                "caption_lexical": hit.caption_lexical,
                "sticker_lexical": hit.sticker_lexical,
            }),
        })
        .collect();
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(top_k);
    Json(SearchResponse { hits })
}

fn build_caption_query(state: &AppState, raw_query: &str, caption_importance: &str, allow_animation: bool) -> Box<dyn Query> {
    let q = raw_query.trim();
    if q.is_empty() {
        return apply_animation_filter(Box::new(AllQuery), &state.schema, allow_animation);
    }
    let mut parser = QueryParser::for_index(
        &state.index,
        vec![
            state.schema.source_overlay_text_normalized,
            state.schema.caption_meaning_en,
            state.schema.caption_meaning_zh,
            state.schema.caption_semantic_text,
        ],
    );
    parser.set_conjunction_by_default();
    match caption_importance {
        "require" => {
            parser.set_field_boost(state.schema.source_overlay_text_normalized, 4.0);
            parser.set_field_boost(state.schema.caption_meaning_zh, 2.6);
            parser.set_field_boost(state.schema.caption_meaning_en, 2.2);
            parser.set_field_boost(state.schema.caption_semantic_text, 2.0);
        }
        "ignore" => {
            parser.set_field_boost(state.schema.source_overlay_text_normalized, 0.5);
            parser.set_field_boost(state.schema.caption_meaning_zh, 0.6);
            parser.set_field_boost(state.schema.caption_meaning_en, 0.7);
            parser.set_field_boost(state.schema.caption_semantic_text, 1.2);
        }
        _ => {
            parser.set_field_boost(state.schema.source_overlay_text_normalized, 2.8);
            parser.set_field_boost(state.schema.caption_meaning_zh, 1.9);
            parser.set_field_boost(state.schema.caption_meaning_en, 1.7);
            parser.set_field_boost(state.schema.caption_semantic_text, 1.9);
        }
    }
    let (query, _) = parser.parse_query_lenient(q);
    apply_animation_filter(query, &state.schema, allow_animation)
}

fn build_sticker_query(state: &AppState, raw_query: &str, allow_animation: bool) -> Box<dyn Query> {
    let q = raw_query.trim();
    if q.is_empty() {
        return apply_animation_filter(Box::new(AllQuery), &state.schema, allow_animation);
    }
    let mut parser = QueryParser::for_index(
        &state.index,
        vec![state.schema.sticker_semantic_text, state.schema.preview_text, state.schema.style_text, state.schema.selection_notes],
    );
    parser.set_conjunction_by_default();
    parser.set_field_boost(state.schema.sticker_semantic_text, 2.4);
    parser.set_field_boost(state.schema.preview_text, 1.4);
    parser.set_field_boost(state.schema.selection_notes, 1.2);
    parser.set_field_boost(state.schema.style_text, 0.7);
    let (query, _) = parser.parse_query_lenient(q);
    apply_animation_filter(query, &state.schema, allow_animation)
}

fn apply_animation_filter(query: Box<dyn Query>, schema: &StickerSchema, allow_animation: bool) -> Box<dyn Query> {
    if allow_animation {
        return query;
    }
    let animated_term = Term::from_field_u64(schema.animated, 1);
    let animated_query = Box::new(TermQuery::new(animated_term, IndexRecordOption::Basic));
    Box::new(BooleanQuery::new(vec![(Occur::Must, query), (Occur::MustNot, animated_query)]))
}

fn collect_hits(
    searcher: &tantivy::Searcher,
    schema: &StickerSchema,
    query: &dyn Query,
    top_k: usize,
    channel: &str,
    merged: &mut HashMap<String, MutableHit>,
) {
    let top_docs = match searcher.search(query, &TopDocs::with_limit(top_k)) {
        Ok(value) => value,
        Err(_) => return,
    };
    for (score, address) in top_docs {
        let doc = match searcher.doc::<TantivyDocument>(address) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let sticker_id = match doc.get_first(schema.sticker_id).and_then(|v| v.as_str()) {
            Some(value) if !value.is_empty() => value.to_string(),
            _ => continue,
        };
        let hit = merged.entry(sticker_id.clone()).or_insert_with(|| MutableHit {
            sticker_id,
            ..MutableHit::default()
        });
        if !hit.fields.iter().any(|value| value == channel) {
            hit.fields.push(channel.to_string());
        }
        match channel {
            "caption_lexical" => hit.caption_lexical = hit.caption_lexical.max(score),
            "sticker_lexical" => hit.sticker_lexical = hit.sticker_lexical.max(score),
            _ => {}
        }
        hit.score = hit.caption_lexical + hit.sticker_lexical;
    }
}
